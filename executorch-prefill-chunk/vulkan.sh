#!/usr/bin/env bash
# The prefill chunk check on ExecuTorch's Vulkan GPU backend, on a Linux x86_64 runner, from pytorch/executorch main
# at MAIN_COMMIT. export_llm allows dynamic shape with Vulkan (it rejects it only with QNN and CoreML), so a Vulkan
# KV-cache export gets builder.py's token bound like an XNNPACK one.
#
# Host (SwiftShader, a CPU Vulkan device, set up as upstream's test-vulkan-models-linux job does):
#   host_*.txt / .json      prefill_check.py on each stock export, main as is (observe)
#   fix_*.txt / .json       after fix.patch: the stock files and the patched export_llm's files, --expect fixed
# Android arm64 (run on phones with device.sh vulkan):
#   bin/llama_main_vk, bin/llama_main_vk_fixed   examples/models/llama runner with the Vulkan backend, main without
#                                                and with fix.patch
#   stories110m_S128_C512_vulkan.pte             stock export_llm, Vulkan, fp32
#   stories110m_S128_C512_vulkan_8da4w.pte       stock export_llm, Vulkan, 8da4w group 64, force_fp16 (the recipe of
#                                                docs/source/backends/vulkan/tutorials/etvk-llama-tutorial.md)
#   stories110m_S512_C512_vulkan.pte             stock export_llm, Vulkan, fp32, S == C
#   tokenizer.model, tokenizer.bin (device.sh pushes both), MANIFEST.txt
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${RUNNER_TEMP:-/tmp}/prefill-vulkan
OUT=$WORK/out
MAIN_COMMIT=${MAIN_COMMIT:-9875560827}
JOBS=$(nproc)
mkdir -p "$WORK" "$OUT/bin"
ET=$WORK/src/executorch
export ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT:-${ANDROID_NDK_LATEST_HOME:?no Android NDK on this runner}}

echo "::group::checkout executorch $MAIN_COMMIT"
git clone -q https://github.com/pytorch/executorch.git "$ET"
git -C "$ET" checkout -q "$MAIN_COMMIT"
git -C "$ET" submodule update --init --recursive -q
git -C "$ET" log -1 --format='%H %cd %s'
echo "::endgroup::"
cd "$ET"

echo "::group::Vulkan SDK (glslc) and SwiftShader, as upstream CI sets them up"
set +u
# shellcheck disable=SC1091
source .ci/scripts/setup-vulkan-linux-deps.sh > "$WORK/vulkan_deps.log" 2>&1
set +x -u
command -v glslc && glslc --version | head -1
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"
echo "::endgroup::"

python -m venv "$WORK/venv"
PY=$WORK/venv/bin/python
export PATH=$WORK/venv/bin:$PATH
"$PY" -m pip install -q --upgrade pip

install_source() {  # install_source <label>: executorch from the checkout, with the Vulkan backend in the pybindings
  echo "::group::python: executorch from source with Vulkan ($1)"
  CMAKE_ARGS="-DEXECUTORCH_BUILD_VULKAN=ON" PYTHON_EXECUTABLE="$PY" ./install_executorch.sh > "$WORK/install_$1.log" 2>&1 \
    || { tail -60 "$WORK/install_$1.log"; exit 1; }
  "$PY" -c "import executorch.version as v; from executorch.extension.pybindings import portable_lib as p; \
print('executorch', v.__version__, v.git_version[:12], p._get_registered_backend_names())"
  echo "::endgroup::"
}

install_source main
"$PY" -m pip install -q sentencepiece tiktoken

echo "::group::stories110m checkpoint and tokenizer"
curl -sSfL -o "$WORK/stories110M.pt" https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt
curl -sSfL -o "$OUT/tokenizer.model" https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model
echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' \
  > "$WORK/params.json"
"$PY" -m pytorch_tokenizers.tools.llama2c.convert -t "$OUT/tokenizer.model" -o "$OUT/tokenizer.bin"
sha256sum "$WORK/stories110M.pt" "$OUT/tokenizer.model"
echo "::endgroup::"

export_vk() {  # export_vk <output.pte> <S> <C> [extra export_llm overrides]
  local pte=$1 s=$2 c=$3; shift 3
  echo "::group::export $(basename "$pte")"
  "$PY" -m executorch.extension.llm.export.export_llm base.model_class=stories110m \
    base.checkpoint="$WORK/stories110M.pt" base.params="$WORK/params.json" \
    model.use_kv_cache=True model.use_sdpa_with_kv_cache=True model.enable_dynamic_shape=True \
    export.max_seq_length="$s" export.max_context_length="$c" backend.vulkan.enabled=True "$@" \
    export.output_name="$pte" > "$WORK/export_$(basename "$pte" .pte).log" 2>&1 \
    || { tail -40 "$WORK/export_$(basename "$pte" .pte).log"; exit 1; }
  ls -la "$pte"
  echo "::endgroup::"
}

check() {  # check <pte> <name> [observe|fixed] [expected bound]
  local pte=$1 name=$2 expect=${3:-observe} bound=${4:-} rc
  echo "::group::prefill_check $name (expect $expect)"
  set +e
  "$PY" "$HERE/prefill_check.py" "$pte" "$OUT/tokenizer.model" --json "$OUT/$name.json" --expect "$expect" \
    ${bound:+--expect-bound "$bound"} > "$OUT/$name.log" 2>&1
  rc=$?
  set -e
  grep -E "^(platform|advertised|module|chunked|runner|prefill|workaround|first runner|expect)" "$OUT/$name.log" \
    | tee "$OUT/$name.txt"
  echo "::endgroup::"
  return $rc
}

V128=$OUT/stories110m_S128_C512_vulkan.pte
V128Q=$OUT/stories110m_S128_C512_vulkan_8da4w.pte
V512=$OUT/stories110m_S512_C512_vulkan.pte
export_vk "$V128" 128 512
export_vk "$V128Q" 128 512 quantization.qmode=8da4w quantization.group_size=64 backend.vulkan.force_fp16=True
export_vk "$V512" 512 512
check "$V128" host_S128_C512_vulkan
check "$V128Q" host_S128_C512_vulkan_8da4w
check "$V512" host_S512_C512_vulkan

android_cmake() {  # android_cmake <build dir>: core libraries with Vulkan and XNNPACK, then the llama runner
  local b=$1
  cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 -DCMAKE_INSTALL_PREFIX="$b" -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
    -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON -DEXECUTORCH_ENABLE_LOGGING=1 -DPYTHON_EXECUTABLE="$PY" \
    -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_LLM=ON -B"$b" . \
  && cmake --build "$b" -j"$JOBS" --target install --config Release \
  && cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 -DCMAKE_INSTALL_PREFIX="$b" -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$PY" \
    -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_LLM=ON -DSUPPORT_REGEX_LOOKAHEAD=ON \
    -B"$b/examples/models/llama" examples/models/llama \
  && cmake --build "$b/examples/models/llama" -j"$JOBS" --config Release
}

echo "::group::android: llama_main (Vulkan), main as is"
android_cmake "$ET/cmake-out-android-vk" > "$WORK/cmake-out-android-vk.log" 2>&1 \
  || { tail -80 "$WORK/cmake-out-android-vk.log"; exit 1; }
cp cmake-out-android-vk/examples/models/llama/llama_main "$OUT/bin/llama_main_vk"
echo "::endgroup::"

git apply "$HERE/fix.patch"
git diff --stat
install_source fixed
rc=0
check "$V128" fix_runner_S128_C512_vulkan fixed 127 || rc=1
check "$V128Q" fix_runner_S128_C512_vulkan_8da4w fixed 127 || rc=1
check "$V512" fix_runner_S512_C512_vulkan fixed 511 || rc=1
P128=$WORK/stories110m_S128_C512_vulkan_patched.pte
P512=$WORK/stories110m_S512_C512_vulkan_patched.pte
export_vk "$P128" 128 512
export_vk "$P512" 512 512
check "$P128" fix_export_S128_C512_vulkan fixed 128 || rc=1
check "$P512" fix_export_S512_C512_vulkan fixed 512 || rc=1

echo "::group::android: llama_main (Vulkan), main + fix.patch"
android_cmake "$ET/cmake-out-android-vk-fixed" > "$WORK/cmake-out-android-vk-fixed.log" 2>&1 \
  || { tail -80 "$WORK/cmake-out-android-vk-fixed.log"; exit 1; }
cp cmake-out-android-vk-fixed/examples/models/llama/llama_main "$OUT/bin/llama_main_vk_fixed"
echo "::endgroup::"
git apply -R "$HERE/fix.patch"

{
  echo "executorch $(git rev-parse HEAD)"
  echo "fix.patch sha256 $(sha256sum "$HERE/fix.patch" | cut -d' ' -f1)"
  echo "NDK $(basename "$ANDROID_NDK_ROOT")"
  echo "glslc $(glslc --version | head -1)"
  (cd "$OUT" && find . -type f \( -name '*.pte' -o -path './bin/*' \) -exec sha256sum {} \; | sort -k2)
} | tee "$OUT/MANIFEST.txt"
exit $rc
