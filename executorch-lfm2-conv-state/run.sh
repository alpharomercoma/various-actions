#!/usr/bin/env bash
# Reproduce pytorch/executorch#23262 and verify the fix in fix.patch on this machine.
#   run.sh unit <runtime>            new regression tests: must fail unpatched, pass patched (+ neighbouring tests)
#   run.sh e2e  <runtime> <model>    export unpatched and patched with the README recipe, then lfm2_state_check.py
# runtime: nightly (executorch 1.6.0.dev20261002 = main 0f5dc8e8) | stable (executorch 1.5.1 from PyPI)
# model:   lfm2_350m | lfm2_5_350m | lfm2_5_1_2b
set -euo pipefail
MODE=$1; RUNTIME=$2; MODEL=${3:-}
HERE=$(cd "$(dirname "$0")" && pwd)
ET_COMMIT=0f5dc8e8d4b97e5425f3efcae2cfe1711fec3a4e
WORK=${RUNNER_TEMP:-/tmp}/lfm2-conv-state
command -v cygpath >/dev/null && WORK=$(cygpath -u "$WORK")
mkdir -p "$WORK"; cd "$WORK"
PY=${PYTHON:-python}

echo "::group::install $RUNTIME"
$PY -m pip install -q --upgrade pip
if [ "$RUNTIME" = nightly ]; then
  $PY -m pip install -q --pre --index-url https://download.pytorch.org/whl/nightly/cpu \
    --extra-index-url https://pypi.org/simple "executorch==1.6.0.dev20261002" "torch==2.15.0.dev20261002" pytest
else
  $PY -m pip install -q --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple "executorch==1.5.1" "torch==2.14.0" pytest
fi
$PY -m pip install -q transformers safetensors huggingface_hub
$PY - <<'EOF'
import torch, executorch.version as v
print("torch", torch.__version__, "| executorch", v.__version__, v.git_version)
EOF
echo "::endgroup::"

echo "::group::executorch sources at $ET_COMMIT + fix.patch"
if [ ! -d et ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/pytorch/executorch.git et
  git -C et sparse-checkout set examples/models/lfm2 examples/models/llama/tests
  git -C et checkout -q $ET_COMMIT
  git -C et apply "$HERE/fix.patch"
fi
git -C et status --short
echo "::endgroup::"

SITE=$($PY -c "import executorch, os; print(list(executorch.__path__)[0])")
command -v cygpath >/dev/null && SITE=$(cygpath -u "$SITE")
SC="$SITE/examples/models/lfm2/short_conv.py"
[ -f "$WORK/short_conv.orig.py" ] || cp "$SC" "$WORK/short_conv.orig.py"
cmp -s "$SC" "$WORK/short_conv.orig.py"
unpatch() { cp "$WORK/short_conv.orig.py" "$SC"; }
patch_() { cp et/examples/models/lfm2/short_conv.py "$SC"; }
# Tests run from a directory without the executorch sources, so they import the installed package.
mkdir -p tests && cp et/examples/models/llama/tests/test_lfm2_short_conv.py tests/

if [ "$MODE" = unit ]; then
  unpatch
  echo "::group::unpatched: new tests must fail"
  set +e; (cd tests && $PY -m pytest -q -p no:cacheprovider test_lfm2_short_conv.py) | tee unpatched.txt; rc=${PIPESTATUS[0]}; set -e
  echo "::endgroup::"
  [ "$rc" -ne 0 ] && grep -q "4 failed" unpatched.txt || { echo "FAIL: expected 4 failures on unpatched code"; exit 1; }
  grep -c "AssertionError: Tensor-likes are not close" unpatched.txt || true
  patch_
  echo "::group::patched: new tests must pass"
  (cd tests && $PY -m pytest -q -p no:cacheprovider test_lfm2_short_conv.py)
  echo "::endgroup::"
  if [ "$RUNTIME" = nightly ]; then
    echo "::group::patched: neighbouring tests from the same commit"
    cp et/examples/models/llama/tests/test_qwen3_5_attention.py et/examples/models/llama/tests/test_transformer_block.py tests/
    (cd tests && $PY -m pytest -q -p no:cacheprovider test_qwen3_5_attention.py test_transformer_block.py)
    echo "::endgroup::"
  fi
  unpatch
  echo "PASS unit $RUNTIME: 4 new tests fail unpatched and pass patched"
  exit 0
fi

case $MODEL in
  lfm2_350m) PARAMS=lfm2_350m_config.json; REPO=LiquidAI/LFM2-350M;;
  lfm2_5_350m) PARAMS=lfm2_5_350m_config.json; REPO=LiquidAI/LFM2.5-350M;;
  lfm2_5_1_2b) PARAMS=lfm2_5_1_2b_config.json; REPO=LiquidAI/LFM2.5-1.2B-Instruct;;
  *) echo "unknown model $MODEL"; exit 2;;
esac
C=et/examples/models/lfm2/config
export_pte() {
  echo "::group::export $1 ($MODEL, lfm2_xnnpack_q8da4w.yaml)"
  $PY -m executorch.extension.llm.export.export_llm --config $C/lfm2_xnnpack_q8da4w.yaml \
    +base.model_class="$MODEL" +base.params="$C/$PARAMS" +export.output_name="$1_$MODEL.pte" > "export_$1.log" 2>&1 \
    || { tail -40 "export_$1.log"; exit 1; }
  ls -la "$1_$MODEL.pte"
  echo "::endgroup::"
}
unpatch; export_pte unpatched
patch_;  export_pte patched
unpatch
TOK=$($PY -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('$REPO', 'tokenizer.json'))")
echo "::group::lfm2_state_check.py"
set +e
$PY "$HERE/lfm2_state_check.py" "unpatched_$MODEL.pte" "patched_$MODEL.pte" "$TOK" --json "result_${RUNTIME}_$MODEL.json" > check.log 2>&1
rc=$?
set -e
echo "::endgroup::"
grep -E "^(platform|sha256|leak |reset |unchanged|PASS|FAIL)" check.log | tee summary.txt
[ $rc -eq 0 ] || { tail -60 check.log; exit $rc; }
