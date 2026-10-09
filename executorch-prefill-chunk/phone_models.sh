#!/usr/bin/env bash
# Phone files for qwen3_0_6b and lfm2_350m, exported with a released executorch wheel and the same stock export_llm
# command as run.sh (weights downloaded from the Hugging Face Hub by export_llm):
#
#   phone_models.sh <python with executorch> <executorch source checkout of that version> <out dir>
#
#   <model>_S128_C512_xnnpack.pte        XNNPACK, fp32, KV cache, dynamic shape (the CI cell)
#   <model>_S128_C512_vulkan_8da4w.pte   Vulkan, 8da4w group 64, force_fp16 (the Vulkan llama tutorial's recipe)
#   <model>.tokenizer.json               the model's Hugging Face tokenizer
#   prompts_<model>/p<N>.txt             prompts of exactly N tokens with that tokenizer (C++ HFTokenizer, no BOS)
#   host_<file>.txt                      prefill_check.py on each file with the same python (observe)
#   MANIFEST.txt
#
# device.sh <out dir> models runs them with llama_main / llama_main_fixed (XNNPACK files) and llama_main_vk /
# llama_main_vk_fixed (Vulkan files); copy those binaries into <out dir>/bin first.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=$1 SRC=$2 OUT=$3
mkdir -p "$OUT"
"$PY" -c "import executorch.version as v, torch; print('executorch', v.__version__, v.git_version[:12], '| torch', torch.__version__)"

params_for() {
  case $1 in
    qwen3_0_6b) echo "$SRC/examples/models/qwen3/config/0_6b_config.json" ;;
    lfm2_350m) echo "$SRC/examples/models/lfm2/config/lfm2_350m_config.json" ;;
  esac
}
hub_for() {
  case $1 in
    qwen3_0_6b) echo Qwen/Qwen3-0.6B ;;
    lfm2_350m) echo LiquidAI/LFM2-350M ;;
  esac
}

for m in qwen3_0_6b lfm2_350m; do
  common=(base.model_class="$m" base.params="$(params_for "$m")" model.use_kv_cache=True
          model.enable_dynamic_shape=True model.use_sdpa_with_kv_cache=True
          export.max_seq_length=128 export.max_context_length=512)
  echo "== export $m XNNPACK fp32"
  (cd "$OUT" && "$PY" -m executorch.extension.llm.export.export_llm "${common[@]}" backend.xnnpack.enabled=True \
    export.output_name="$OUT/${m}_S128_C512_xnnpack.pte") > "$OUT/export_${m}_xnnpack.log" 2>&1 \
    || { tail -30 "$OUT/export_${m}_xnnpack.log"; exit 1; }
  echo "== export $m Vulkan 8da4w fp16"
  (cd "$OUT" && "$PY" -m executorch.extension.llm.export.export_llm "${common[@]}" backend.vulkan.enabled=True \
    backend.vulkan.force_fp16=True quantization.qmode=8da4w quantization.group_size=64 \
    export.output_name="$OUT/${m}_S128_C512_vulkan_8da4w.pte") > "$OUT/export_${m}_vulkan.log" 2>&1 \
    || { tail -30 "$OUT/export_${m}_vulkan.log"; exit 1; }
  tok=$("$PY" -c "from huggingface_hub import hf_hub_download as d; print(d('$(hub_for "$m")', 'tokenizer.json'))")
  cp "$tok" "$OUT/$m.tokenizer.json"
  mkdir -p "$OUT/prompts_$m"
  "$PY" - "$OUT/$m.tokenizer.json" "$OUT/prompts_$m" <<'EOF'
import sys
from pytorch_tokenizers import CppHFTokenizer
tok = CppHFTokenizer(); tok.load(sys.argv[1])
for n in (127, 128, 129, 257):
    text = " apple" * n
    ids = tok.encode(text, 0, 0)
    assert len(ids) == n, (n, len(ids))
    open(f"{sys.argv[2]}/p{n}.txt", "w").write(text)
print("prompts", sys.argv[2], "ok")
EOF
  # Host check of the XNNPACK file (the host wheel may not have the Vulkan runtime).
  "$PY" "$HERE/prefill_check.py" "$OUT/${m}_S128_C512_xnnpack.pte" "$OUT/$m.tokenizer.json" \
    --json "$OUT/host_${m}_xnnpack.json" 2>/dev/null \
    | grep -E "^(platform|advertised|module|runner|prefill|workaround)" | tee "$OUT/host_${m}_xnnpack.txt"
  "$PY" - "$OUT/${m}_S128_C512_vulkan_8da4w.pte" <<'EOF' | tee "$OUT/host_${m}_vulkan.txt"
import sys
from executorch.runtime import Runtime
# Program metadata only: reading it does not initialize the Vulkan delegate.
bound = Runtime.get().load_program(sys.argv[1]).metadata("forward").input_tensor_meta(0).sizes()[1]
b = open(sys.argv[1], "rb").read()
print("vulkan file:", sys.argv[1].rsplit("/", 1)[-1], "| forward token bound", bound, "| VulkanBackend",
      b.count(b"VulkanBackend"), "| XnnpackBackend", b.count(b"XnnpackBackend"))
EOF
done
(cd "$OUT" && shasum -a 256 ./*.pte ./*.tokenizer.json) | tee "$OUT/MANIFEST.txt"
