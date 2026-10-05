#!/usr/bin/env bash
# Approximate the internal Buck test for #23401: build a link tree with only the BUCK dependency closure of
# examples/models/llama/tests:test_lfm2_short_conv and run it with torch but WITHOUT executorch installed.
#   fix_v1.patch (first PR commit): must fail with ModuleNotFoundError executorch.examples.models.checkpoint
#   fix.patch (current PR): must pass 4/4
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ET_COMMIT=0f5dc8e8d4b97e5425f3efcae2cfe1711fec3a4e
WORK=${RUNNER_TEMP:-/tmp}/lfm2-buck-closure
command -v cygpath >/dev/null && WORK=$(cygpath -u "$WORK")
mkdir -p "$WORK"; cd "$WORK"
PY=${PYTHON:-python}
$PY -m pip install -q --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
  "torch==2.14.0" numpy safetensors
$PY -c "import importlib.util as u, torch; print('torch', torch.__version__); assert u.find_spec('executorch') is None"
run_case() {  # name patch expect(pass|checkpoint_missing)
  rm -rf "et_$1" "lt_$1"
  git clone -q --filter=blob:none --no-checkout https://github.com/pytorch/executorch.git "et_$1"
  git -C "et_$1" sparse-checkout set examples/models
  git -C "et_$1" checkout -q $ET_COMMIT
  git -C "et_$1" apply "$HERE/$2"
  $PY "$HERE/buck_linktree.py" "et_$1" examples/models/llama/tests:test_lfm2_short_conv "lt_$1"
  set +e
  (cd "lt_$1" && $PY -m unittest -v executorch.examples.models.llama.tests.test_lfm2_short_conv) > "test_$1.log" 2>&1
  rc=$?
  set -e
  grep -E "ModuleNotFoundError|^Ran|^OK|^FAILED|\.\.\. (ok|FAIL|ERROR)" "test_$1.log" || true
  if [ "$3" = pass ]; then [ $rc -eq 0 ] && grep -q "Ran 4 tests" "test_$1.log"
  else [ $rc -ne 0 ] && grep -q "No module named 'executorch.examples.models.checkpoint'" "test_$1.log"; fi
  echo "PASS $1: expected $3"
}
run_case v1 fix_v1.patch checkpoint_missing
run_case current fix.patch pass
