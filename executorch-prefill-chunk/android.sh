#!/usr/bin/env bash
# Build everything the on-device runs need, on a Linux x86_64 runner, from pytorch/executorch main at MAIN_COMMIT:
#
#   bin/llama_main          examples/models/llama runner (TextLLMRunner, XNNPACK), Android arm64, main as is
#   bin/llama_main_fixed    the same runner with fix.patch applied
#   bin/qnn_llama_runner    examples/qualcomm/oss_scripts/llama runner (its own prompt processor), Android arm64
#   qnn/                    the QNN runtime libraries the runner loads (aarch64-android, HTP v81 skel for SM8850)
#   stories110m_S128_C512_xnnpack.pte   stock export_llm export (KV cache, dynamic shape, XNNPACK), fp32
#   stories110m_qnn_SM8850/  llama.py export for SM8850: hybrid mode, prefill_ar_len 32, context 512, 16a4w
#   tokenizer.model, tokenizer.bin
#
# The XNNPACK file must fail with prompts of 128 tokens or more on llama_main and succeed on llama_main_fixed; the QNN
# file is the control (static shapes, the runner reads its chunk from the program).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${RUNNER_TEMP:-/tmp}/prefill-android
OUT=$WORK/out
MAIN_COMMIT=${MAIN_COMMIT:-9875560827}
QNN_VERSION=${QNN_VERSION:-2.42.0.251225}
JOBS=$(nproc)
mkdir -p "$WORK" "$OUT/bin" "$OUT/qnn"
ET=$WORK/src/executorch
export ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-${ANDROID_NDK_LATEST_HOME:?no Android NDK on this runner}}

echo "::group::checkout executorch $MAIN_COMMIT"
git clone -q https://github.com/pytorch/executorch.git "$ET"
git -C "$ET" checkout -q "$MAIN_COMMIT"
git -C "$ET" submodule update --init --recursive -q
git -C "$ET" log -1 --format='%H %cd %s'
echo "::endgroup::"
cd "$ET"

echo "::group::QNN SDK $QNN_VERSION"
# The version install_qnn_sdk.sh pins (2.37) has no HTP v81 libraries, which SM8850 needs; use a newer SDK from the
# same public location.
curl -sSfL -o "$WORK/qairt.zip" \
  "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/$QNN_VERSION/v$QNN_VERSION.zip"
unzip -q "$WORK/qairt.zip" -d "$WORK/qairt-unzip" && rm "$WORK/qairt.zip"
QNN_SDK_ROOT=$(find "$WORK/qairt-unzip" -maxdepth 3 -type d -name "$QNN_VERSION" | head -1)
export QNN_SDK_ROOT
ls -la "$QNN_SDK_ROOT"/lib/aarch64-android/libQnnHtp.so "$QNN_SDK_ROOT"/lib/aarch64-android/libQnnHtpV81Stub.so
echo "QNN_SDK_ROOT=$QNN_SDK_ROOT"
echo "::endgroup::"

echo "::group::python: executorch from source (with the Qualcomm AOT backend, since QNN_SDK_ROOT is set)"
python -m venv "$WORK/venv"
PY=$WORK/venv/bin/python
export PATH=$WORK/venv/bin:$PATH
"$PY" -m pip install -q --upgrade pip
PYTHON_EXECUTABLE="$PY" ./install_executorch.sh > "$WORK/install.log" 2>&1 || { tail -60 "$WORK/install.log"; exit 1; }
"$PY" -m pip install -q -r backends/qualcomm/requirements.txt lm_eval==0.4.5 sentencepiece tiktoken
"$PY" -c "import executorch.version as v, torch; print('executorch', v.__version__, v.git_version[:12], '| torch', torch.__version__)"
echo "::endgroup::"

echo "::group::stories110m checkpoint and tokenizers"
curl -sSfL -o "$WORK/stories110M.pt" https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt
curl -sSfL -o "$OUT/tokenizer.model" https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' \
  > "$WORK/params.json"
"$PY" -m pytorch_tokenizers.tools.llama2c.convert -t "$OUT/tokenizer.model" -o "$OUT/tokenizer.bin"
sha256sum "$WORK/stories110M.pt" "$OUT/tokenizer.model"
echo "::endgroup::"

echo "::group::export: stock export_llm, XNNPACK, S=128 C=512"
"$PY" -m executorch.extension.llm.export.export_llm base.model_class=stories110m \
  base.checkpoint="$WORK/stories110M.pt" base.params="$WORK/params.json" \
  model.use_kv_cache=True model.use_sdpa_with_kv_cache=True model.enable_dynamic_shape=True \
  export.max_seq_length=128 export.max_context_length=512 backend.xnnpack.enabled=True \
  export.output_name="$OUT/stories110m_S128_C512_xnnpack.pte" > "$WORK/export_xnnpack.log" 2>&1 \
  || { tail -40 "$WORK/export_xnnpack.log"; exit 1; }
"$PY" "$HERE/prefill_check.py" "$OUT/stories110m_S128_C512_xnnpack.pte" "$OUT/tokenizer.model" \
  --json "$OUT/host_check_xnnpack.json" 2>/dev/null | tee "$OUT/host_check_xnnpack.txt"
echo "::endgroup::"

echo "::group::android: QNN runtime and qnn_llama_runner (the cmake steps of backends/qualcomm/scripts/build.sh)"
# build.sh builds every Qualcomm example, and at this commit qaihub_llama3_8b_runner does not link
# (undefined example::get_tiktoken_for_llama), so build the same configuration but only qnn_llama_runner.
B=$ET/build-android
{
  cmake -S . -B "$B" -DCMAKE_INSTALL_PREFIX="$B" -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_QNN=ON \
    -DQNN_SDK_ROOT="$QNN_SDK_ROOT" -DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DANDROID_PLATFORM=android-30 -DPYTHON_EXECUTABLE="$PY" \
  && cmake --build "$B" -j"$JOBS" --target install \
  && cmake -S examples/qualcomm -B "$B/examples/qualcomm" \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DCMAKE_PREFIX_PATH="$B;$B/third-party/gflags;" \
    -DSUPPORT_REGEX_LOOKAHEAD=ON -DBUILD_TESTING=OFF -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH -DPYTHON_EXECUTABLE="$PY" \
    -DBUILD_DIRECT_MODE=OFF \
  && cmake --build "$B/examples/qualcomm" -j"$JOBS" --target qnn_llama_runner
} > "$WORK/build_qnn.log" 2>&1 || { tail -80 "$WORK/build_qnn.log"; exit 1; }
cp "$(find "$B/examples/qualcomm" -name qnn_llama_runner -type f -perm -u+x | head -1)" "$OUT/bin/"
find build-android -name 'libqnn_executorch_backend.so' -exec cp {} "$OUT/qnn/" \; -quit
for f in libQnnHtp.so libQnnHtpPrepare.so libQnnSystem.so libQnnHtpV81Stub.so; do
  cp "$QNN_SDK_ROOT/lib/aarch64-android/$f" "$OUT/qnn/"
done
cp "$QNN_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so" "$OUT/qnn/"
ls -la "$OUT/qnn"
echo "::endgroup::"

echo "::group::export: QNN static llama for SM8850 (examples/qualcomm/oss_scripts/llama/llama.py)"
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
"$PY" examples/qualcomm/oss_scripts/llama/llama.py --artifact "$OUT/stories110m_qnn_SM8850" \
  --build_folder build-android --checkpoint "$WORK/stories110M.pt" --params "$WORK/params.json" \
  --tokenizer_model "$OUT/tokenizer.model" --tokenizer_bin "$OUT/tokenizer.bin" --prompt Once --temperature 0 \
  --decoder_model stories110m --model_mode hybrid --prefill_ar_len 32 --max_seq_len 512 --max_context_len 512 \
  --calib_tasks wikitext --calib_limit 1 --soc_model SM8850 --compile_only > "$WORK/export_qnn.log" 2>&1 \
  || { tail -60 "$WORK/export_qnn.log"; exit 1; }
find "$OUT/stories110m_qnn_SM8850" -maxdepth 1 -type f -name '*.pte' -exec ls -la {} \;
echo "::endgroup::"

android_cmake() {  # android_cmake <build dir>: core libraries with XNNPACK, then the llama runner
  # Chained with &&: set -e does not apply inside a function called from an || list.
  local b=$1
  cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 -DCMAKE_INSTALL_PREFIX="$b" -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON -DEXECUTORCH_ENABLE_LOGGING=1 -DPYTHON_EXECUTABLE="$PY" \
    -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON -B"$b" . \
  && cmake --build "$b" -j"$JOBS" --target install --config Release \
  && cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-23 -DCMAKE_INSTALL_PREFIX="$b" -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$PY" \
    -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_LLM=ON -DSUPPORT_REGEX_LOOKAHEAD=ON -B"$b/examples/models/llama" examples/models/llama \
  && cmake --build "$b/examples/models/llama" -j"$JOBS" --config Release
}

echo "::group::android: llama_main (XNNPACK), main as is"
android_cmake "$ET/cmake-out-android" > "$WORK/cmake-out-android.log" 2>&1 || { tail -80 "$WORK/cmake-out-android.log"; exit 1; }
cp cmake-out-android/examples/models/llama/llama_main "$OUT/bin/llama_main"
echo "::endgroup::"

echo "::group::android: llama_main (XNNPACK), main + fix.patch"
git apply "$HERE/fix.patch"
git diff --stat
android_cmake "$ET/cmake-out-android-fixed" > "$WORK/cmake-out-android-fixed.log" 2>&1 || { tail -80 "$WORK/cmake-out-android-fixed.log"; exit 1; }
cp cmake-out-android-fixed/examples/models/llama/llama_main "$OUT/bin/llama_main_fixed"
git apply -R "$HERE/fix.patch"
echo "::endgroup::"

{
  echo "executorch $(git rev-parse HEAD)"
  echo "fix.patch sha256 $(sha256sum "$HERE/fix.patch" | cut -d' ' -f1)"
  echo "QNN SDK $(basename "$QNN_SDK_ROOT")"
  echo "NDK $(basename "$ANDROID_NDK_ROOT")"
  (cd "$OUT" && find . -type f \( -name '*.pte' -o -path './bin/*' -o -path './qnn/*' \) -exec sha256sum {} \; | sort -k2)
} | tee "$OUT/MANIFEST.txt"
