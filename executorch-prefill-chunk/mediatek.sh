#!/usr/bin/env bash
# MediaTek NeuroPilot control for the prefill chunk check, on a Linux x86_64 runner, from pytorch/executorch main at
# MAIN_COMMIT. MediaTek's llama runner (examples/mediatek, mtk_llama_executor_runner) does not use export_llm's
# dynamic shapes or TextPrefiller: its prompt model has a static token batch, and the runner prefills in batches of it.
# This builds that path for the Dimensity 9400 (MT6991, platform DX4), as upstream CI and examples/mediatek/README.md
# do, to show on a phone that prompts at and above the batch size are prefilled in batches.
#
#   qwen2_5_0_5b_A16W4_128t512c_1t512c.pte  examples/mediatek export of Qwen2.5-0.5B-Instruct: prompt method (128 tokens
#                                        per call) and gen method (1), cache 512, A16W4, DX4, 1 chunk, no weight sharing
#   embedding_*_fp32.bin, tokenizer.json
#   bin/mtk_llama_executor_runner, bin/mtk_llama_executor_runner_fixed   main without and with fix.patch
#   lib/libneuron_backend.so (main), lib/libneuronusdk_adapter.mtk.so, lib/libneuron_buffer_allocator.so
#   prompts/q<N>.txt                     prompts of exactly N tokens with the Qwen tokenizer (no special tokens)
#   MANIFEST.txt
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${RUNNER_TEMP:-/tmp}/prefill-mediatek
OUT=$WORK/out
MAIN_COMMIT=${MAIN_COMMIT:-9875560827}
MODEL=Qwen2.5-0.5B-Instruct
mkdir -p "$WORK" "$OUT/bin" "$OUT/lib" "$OUT/prompts"
ET=$WORK/src/executorch
export ANDROID_NDK=${ANDROID_NDK_ROOT:-${ANDROID_NDK_LATEST_HOME:?no Android NDK on this runner}}

echo "::group::checkout executorch $MAIN_COMMIT"
git clone -q https://github.com/pytorch/executorch.git "$ET"
git -C "$ET" checkout -q "$MAIN_COMMIT"
git -C "$ET" submodule update --init --recursive -q
git -C "$ET" log -1 --format='%H %cd %s'
echo "::endgroup::"
cd "$ET"

echo "::group::python 3.10 (the NeuroPilot converter wheel is cp310): executorch from source"
python3.10 -m venv "$WORK/venv"
PY=$WORK/venv/bin/python
export PATH=$WORK/venv/bin:$PATH
"$PY" -m pip install -q --upgrade pip
PYTHON_EXECUTABLE="$PY" ./install_executorch.sh > "$WORK/install.log" 2>&1 || { tail -60 "$WORK/install.log"; exit 1; }
"$PY" -c "import executorch.version as v, torch; print('executorch', v.__version__, v.git_version[:12], '| torch', torch.__version__)"
echo "::endgroup::"

echo "::group::NeuroPilot Express SDK (.ci/scripts/setup-mediatek-deps.sh)"
bash .ci/scripts/setup-mediatek-deps.sh > "$WORK/mediatek_deps.log" 2>&1 || { tail -40 "$WORK/mediatek_deps.log"; exit 1; }
# examples/mediatek's vendored tokenizer code imports transformers 4.x names (SpecialTokensMixin), which transformers 5
# removed.
"$PY" -m pip install -q "transformers<5"
export MEDIATEK_SDK_ROOT=/tmp/neuropilot
export NEURON_BUFFER_ALLOCATOR_LIB=$MEDIATEK_SDK_ROOT/libneuron_buffer_allocator.so
"$PY" -m pip list 2>/dev/null | grep -i -E "mtk|torch |transformers"
ls "$MEDIATEK_SDK_ROOT"
echo "::endgroup::"

echo "::group::$MODEL weights (config.json is examples/mediatek's own)"
W=examples/mediatek/models/llm_models/weights/$MODEL
"$PY" - "$W" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir=sys.argv[1],
                  allow_patterns=["*.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                                  "merges.txt", "generation_config.json"])
EOF
git checkout -- "$W/config.json"
ls -la "$W"
echo "::endgroup::"

echo "::group::export: examples/mediatek qwen, 1 chunk, 128t512c + 1t512c, A16W4, DX4, without weight sharing"
# Weight sharing between the prompt and gen methods (ExtractSharedBlobKey, #13941) needs mtk_neuron.extract_shared_data,
# which the public NeuroPilot Express SDK that upstream CI installs (mtk_neuron 8.2.19) does not have. Without the key,
# each method keeps its own weights (NeuronBackend only shares weights when the key is in its compile specs); the
# shapes are still exported together, as export_qwen.sh does (exporting 1t512c alone specializes the token dim).
Q=examples/mediatek/model_export_scripts/qwen.py
grep -q 'CompileSpec("ExtractSharedBlobKey"' "$Q"
sed -i '/CompileSpec("ExtractSharedBlobKey"/d' "$Q"
if grep -q 'ExtractSharedBlobKey' "$Q"; then echo "ExtractSharedBlobKey still in $Q"; exit 1; fi
(cd examples/mediatek && bash shell_scripts/export_qwen.sh "$MODEL" 1 128 512 None A16W4 DX4) > "$WORK/export.log" 2>&1 \
  || { tail -60 "$WORK/export.log"; exit 1; }
cp "$(find examples/mediatek/pte -name '*.pte' | head -1)" "$OUT/qwen2_5_0_5b_A16W4_128t512c_1t512c.pte"
cp "$W"/embedding_*_fp32.bin "$W/tokenizer.json" "$OUT/"
ls -la "$OUT"/*.pte "$OUT"/embedding_*_fp32.bin
echo "::endgroup::"

echo "::group::prompts of exactly N Qwen tokens"
"$PY" - "$W/tokenizer.json" "$OUT/prompts" <<'EOF'
import sys
from pytorch_tokenizers import CppHFTokenizer
tok = CppHFTokenizer(); tok.load(sys.argv[1])
for n in (126, 127, 128, 129, 255, 256, 257):
    text = " apple" * n
    ids = tok.encode(text, 0, 0)
    assert len(ids) == n, (n, len(ids))
    open(f"{sys.argv[2]}/q{n}.txt", "w").write(text)
    print(n, "tokens ok")
EOF
echo "::endgroup::"

build_mtk() {  # build_mtk <suffix>: backend and examples (backends/mediatek/scripts/mtk_build.sh, mtk_build_examples.sh)
  ./backends/mediatek/scripts/mtk_build.sh && ./examples/mediatek/mtk_build_examples.sh \
  && cp cmake-android-out/examples/mediatek/mtk_llama_executor_runner "$OUT/bin/mtk_llama_executor_runner$1"
}

echo "::group::android: MediaTek backend and mtk_llama_executor_runner, main as is"
build_mtk "" > "$WORK/build_mtk.log" 2>&1 || { tail -80 "$WORK/build_mtk.log"; exit 1; }
cp "$(find cmake-android-out -name libneuron_backend.so | head -1)" "$OUT/lib/"
cp "$MEDIATEK_SDK_ROOT/libneuron_buffer_allocator.so" "$OUT/lib/"
cp "$(find "$MEDIATEK_SDK_ROOT" -name 'libneuronusdk_adapter.mtk.so' | head -1)" "$OUT/lib/"
echo "::endgroup::"

echo "::group::android: mtk_llama_executor_runner, main + fix.patch"
git apply "$HERE/fix.patch"
git diff --stat
build_mtk "_fixed" > "$WORK/build_mtk_fixed.log" 2>&1 || { tail -80 "$WORK/build_mtk_fixed.log"; exit 1; }
git apply -R "$HERE/fix.patch"
echo "::endgroup::"

{
  echo "executorch $(git rev-parse HEAD)"
  echo "fix.patch sha256 $(sha256sum "$HERE/fix.patch" | cut -d' ' -f1)"
  echo "NDK $(basename "$ANDROID_NDK")"
  "$PY" -m pip list 2>/dev/null | grep -i -E "^mtk" || true
  (cd "$OUT" && find . -type f \( -name '*.pte' -o -name '*.bin' -o -path './bin/*' -o -path './lib/*' \) \
    -exec sha256sum {} \; | sort -k2)
} | tee "$OUT/MANIFEST.txt"
