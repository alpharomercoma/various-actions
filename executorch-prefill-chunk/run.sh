#!/usr/bin/env bash
# Export an LLM with stock export_llm and check the prefill chunk contract (prefill_check.py).
#
#   run.sh repro <version> <model> <S> <C> <quant>      install <version>, export, check
#   run.sh cross <export_version> <run_version> <model> <S> <C> <quant>
#                                                     export with one release, check with another
#   run.sh fix <model> <S> <C> <quant>                  build main + fix.patch from source, export, check
#   run.sh unit                                         build main + fix.patch, then the runner's C++ test_runner
#
# version: 1.0.1 | 1.1.0 | 1.2.0 | 1.3.1 | 1.4.1 | 1.5.1 | nightly | source (fix mode only)
# model:   stories110m (Llama) | qwen3_0_6b | lfm2_350m (hybrid conv/attention)
# S, C:    export.max_seq_length (prefill chunk) and export.max_context_length (KV cache)
# quant:   fp32 (none) | 8da4w (quantization.qmode=8da4w, group size 32)
# BACKEND=xnnpack (default) | mlx (Apple Silicon; export_llm's MLX path, which goes through the same builder)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${RUNNER_TEMP:-/tmp}/prefill-chunk
mkdir -p "$WORK"
PY_BASE=${PYTHON:-python3}
BACKEND=${BACKEND:-xnnpack}
NIGHTLY_ET=1.6.0.dev20261008
NIGHTLY_TORCH=2.16.0.dev20261008
# main at the nightly above; the fix is applied on top of it in fix mode.
MAIN_COMMIT=${MAIN_COMMIT:-9875560827}

venv_for() {  # venv_for <version> -> prints the venv's python, installing it on first use
  local v=$1 dir="$WORK/venv-$1"
  if [ ! -x "$dir/bin/python" ]; then
    echo "::group::install executorch $v" >&2
    "$PY_BASE" -m venv "$dir" >&2
    local pip=("$dir/bin/python" -m pip install -q)
    "${pip[@]}" --upgrade pip >&2
    local cpu=(--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple)
    case $v in
      1.0.1) "${pip[@]}" "${cpu[@]}" executorch==1.0.1 "torch==2.9.*" torchao==0.14.0 >&2 ;;
      1.1.0) "${pip[@]}" "${cpu[@]}" executorch==1.1.0 "torch==2.10.*" torchao==0.15.0 >&2 ;;
      1.2.0) "${pip[@]}" "${cpu[@]}" executorch==1.2.0 "torch==2.11.*" torchao==0.17.0 >&2 ;;
      1.3.1) "${pip[@]}" "${cpu[@]}" executorch==1.3.1 "torch==2.12.*" torchao==0.17.0 >&2 ;;
      1.4.1) "${pip[@]}" "${cpu[@]}" executorch==1.4.1 "torch==2.13.*" torchao==0.18.0 >&2 ;;
      1.5.1) "${pip[@]}" "${cpu[@]}" executorch==1.5.1 "torch==2.14.*" torchao==0.18.0 >&2 ;;
      nightly)
        "${pip[@]}" --pre --index-url https://download.pytorch.org/whl/nightly/cpu --extra-index-url https://pypi.org/simple \
          "executorch==$NIGHTLY_ET" "torch==$NIGHTLY_TORCH" >&2 ;;
      *) echo "unknown version $v" >&2; exit 2 ;;
    esac
    "${pip[@]}" transformers safetensors huggingface_hub sentencepiece tiktoken >&2
    # Before 1.2 the qwen3 and lfm2 checkpoint converters import torchtune.
    # torchtune's own pins conflict with these torch releases; its converters need only omegaconf and datasets.
    case $v in 1.0.1|1.1.0) "${pip[@]}" --no-deps torchtune==0.6.1 >&2; "${pip[@]}" omegaconf datasets >&2 ;; esac
    "$dir/bin/python" -c "import torch, executorch.version as v; print('executorch', v.__version__, v.git_version[:12], '| torch', torch.__version__)" >&2
    echo "::endgroup::" >&2
  fi
  echo "$dir/bin/python"
}

sources_for() {  # model configs from the matching tag (export_llm reads params files from the source tree)
  local v=$1 ref dir="$WORK/src-$1"
  case $v in nightly|source) ref=$MAIN_COMMIT ;; *) ref=v$v ;; esac
  if [ ! -d "$dir" ]; then
    git clone -q --filter=blob:none --no-checkout https://github.com/pytorch/executorch.git "$dir"
    git -C "$dir" sparse-checkout set examples/models/qwen3 examples/models/lfm2 >&2
    git -C "$dir" checkout -q "$ref" >&2
  fi
  echo "$dir"
}

tokenizer_for() {  # tokenizer_for <python> <model>
  local py=$1 m=$2
  case $m in
    stories110m)
      [ -f "$WORK/tokenizer.model" ] || curl -sSfL -o "$WORK/tokenizer.model" \
        https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model
      echo "$WORK/tokenizer.model" ;;
    qwen3_0_6b) "$py" -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen3-0.6B', 'tokenizer.json'))" ;;
    lfm2_350m) "$py" -c "from huggingface_hub import hf_hub_download as d; print(d('LiquidAI/LFM2-350M', 'tokenizer.json'))" ;;
  esac
}

export_pte() {  # export_pte <python> <version> <model> <S> <C> <quant> <out.pte>
  local py=$1 v=$2 m=$3 s=$4 c=$5 q=$6 out=$7 src args
  src=$(sources_for "$v")
  args=(base.model_class="$m" model.use_kv_cache=True model.enable_dynamic_shape=True
        export.max_seq_length="$s" export.max_context_length="$c" export.output_name="$out")
  case $BACKEND in
    xnnpack) args+=(model.use_sdpa_with_kv_cache=True backend.xnnpack.enabled=True) ;;
    mlx) args+=(backend.mlx.enabled=True) ;;
    *) echo "unknown backend $BACKEND"; exit 2 ;;
  esac
  case $m in
    stories110m)
      [ -f "$WORK/stories110M.pt" ] || curl -sSfL -o "$WORK/stories110M.pt" \
        https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.pt
      echo '{"dim": 768, "multiple_of": 32, "n_heads": 12, "n_layers": 12, "norm_eps": 1e-05, "vocab_size": 32000}' \
        > "$WORK/stories110M_params.json"
      args+=(base.checkpoint="$WORK/stories110M.pt" base.params="$WORK/stories110M_params.json") ;;
    qwen3_0_6b) args+=(base.params="$src/examples/models/qwen3/config/0_6b_config.json") ;;
    lfm2_350m) args+=(base.params="$src/examples/models/lfm2/config/lfm2_350m_config.json") ;;
  esac
  case $q in
    fp32) ;;
    8da4w) args+=(quantization.qmode=8da4w quantization.group_size=32) ;;
    *) echo "unknown quant $q"; exit 2 ;;
  esac
  echo "::group::export $m S=$s C=$c $q $BACKEND with executorch $v"
  local log="${out%.pte}.export.log"
  (cd "$WORK" && "$py" -m executorch.extension.llm.export.export_llm "${args[@]}") > "$log" 2>&1 \
    || { tail -40 "$log"; exit 1; }
  ls -la "$out"
  echo "::endgroup::"
}

check() {  # check <python> <pte> <tokenizer> <json>
  local py=$1 pte=$2 tok=$3 json=$4
  echo "::group::prefill_check $(basename "$pte")"
  set +e
  "$py" "$HERE/prefill_check.py" "$pte" "$tok" --json "$json" > "${json%.json}.log" 2>&1
  local rc=$?
  set -e
  echo "::endgroup::"
  grep -E "^(platform|file|advertised|module|chunked|runner|workaround|first runner)" "${json%.json}.log" | tee -a "$WORK/summary.txt"
  echo >> "$WORK/summary.txt"
  [ $rc -eq 0 ] || { tail -40 "${json%.json}.log"; exit $rc; }
}

build_source() {  # main + fix.patch, installed into venv-source (python bindings) and kept for the C++ build
  PY="$WORK/venv-source/bin/python"
  if [ ! -x "$PY" ]; then
    echo "::group::build executorch $MAIN_COMMIT + fix.patch from source"
    "$PY_BASE" -m venv "$WORK/venv-source"
    git clone -q https://github.com/pytorch/executorch.git "$WORK/et-main"
    git -C "$WORK/et-main" checkout -q "$MAIN_COMMIT"
    git -C "$WORK/et-main" apply "$HERE/fix.patch"
    git -C "$WORK/et-main" submodule update --init --recursive -q
    (cd "$WORK/et-main" && "$PY" -m pip install -q --upgrade pip && PYTHON_EXECUTABLE="$PY" ./install_executorch.sh) \
      > "$WORK/build.log" 2>&1 || { tail -60 "$WORK/build.log"; exit 1; }
    "$PY" -m pip install -q transformers safetensors huggingface_hub sentencepiece tiktoken
    "$PY" -c "import executorch.version as v; print('built executorch', v.__version__, v.git_version[:12])"
    echo "::endgroup::"
  fi
}

MODE=$1; shift
case $MODE in
  repro)
    V=$1 M=$2 S=$3 C=$4 Q=$5
    PY=$(venv_for "$V")
    TAG=${M}_S${S}_C${C}_${Q}; [ "$BACKEND" = xnnpack ] || TAG=${TAG}_$BACKEND
    PTE="$WORK/${TAG}_et${V}.pte"
    export_pte "$PY" "$V" "$M" "$S" "$C" "$Q" "$PTE"
    check "$PY" "$PTE" "$(tokenizer_for "$PY" "$M")" "$WORK/result_repro_${V}_${TAG}.json" ;;
  cross)
    EV=$1 RV=$2 M=$3 S=$4 C=$5 Q=$6
    EPY=$(venv_for "$EV"); RPY=$(venv_for "$RV")
    PTE="$WORK/${M}_S${S}_C${C}_${Q}_et${EV}.pte"
    [ -f "$PTE" ] || export_pte "$EPY" "$EV" "$M" "$S" "$C" "$Q" "$PTE"
    check "$RPY" "$PTE" "$(tokenizer_for "$RPY" "$M")" "$WORK/result_cross_export${EV}_run${RV}_${M}_S${S}_C${C}_${Q}.json" ;;
  fix)
    M=$1 S=$2 C=$3 Q=$4
    build_source
    PTE="$WORK/${M}_S${S}_C${C}_${Q}_etsource.pte"
    export_pte "$PY" source "$M" "$S" "$C" "$Q" "$PTE"
    check "$PY" "$PTE" "$(tokenizer_for "$PY" "$M")" "$WORK/result_fix_${M}_S${S}_C${C}_${Q}.json" ;;
  unit)
    build_source
    echo "::group::cmake: extension/llm/runner tests (test_runner)"
    B="$WORK/cmake-tests"
    (cd "$WORK/et-main" && cmake . -B "$B" -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$PY" \
      -DEXECUTORCH_BUILD_TESTS=ON -DEXECUTORCH_BUILD_KERNELS_LLM=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
      -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_IMAGE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
      -DEXECUTORCH_BUILD_EXTENSION_LLM=ON -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
      -DEXECUTORCH_BUILD_XNNPACK=ON > "$WORK/cmake-configure.log" 2>&1) || { tail -60 "$WORK/cmake-configure.log"; exit 1; }
    cmake --build "$B" --target test_runner -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)" \
      > "$WORK/cmake-build.log" 2>&1 || { tail -80 "$WORK/cmake-build.log"; exit 1; }
    echo "::endgroup::"
    (cd "$B" && ctest -R '^test_runner$' --output-on-failure) | tee "$WORK/ctest.log"
    # Show the new tests by name (ctest prints only the binary's verdict): same fixtures, filtered run.
    T="$B/extension/llm/runner/test"
    ET_PREFILL_CHUNK_BOUNDED_PATH="$T/PrefillChunk_bounded.pte" ET_PREFILL_CHUNK_FULL_PATH="$T/PrefillChunk_full.pte" \
      "$T/test_runner" --gtest_filter='PrefillChunkSizeTest.*' | tee -a "$WORK/ctest.log"
    grep -E "^\[ +(OK|FAILED|PASSED) +\]|tests passed|tests failed" "$WORK/ctest.log" | tee -a "$WORK/summary.txt" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
