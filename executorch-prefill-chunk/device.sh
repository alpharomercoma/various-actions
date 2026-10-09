#!/usr/bin/env bash
# Run the on-device checks with the files from the executorch-prefill-chunk-android workflow (its `prefill-android`
# artifact, unpacked into <dir>), over adb:
#
#   device.sh <dir> xnnpack   llama_main and llama_main_fixed on the stock export_llm XNNPACK export (S=128, C=512):
#                             prompts of exactly 127, 128, 129 and 257 tokens
#   device.sh <dir> qnn       qnn_llama_runner on the QNN export for SM8850 (prefill_ar_len 32, context 512):
#                             prompts of exactly 31, 32, 33, 65, 128, 129 and 257 tokens
#
# Prompts are " apple" repeated (prompts/p<N>.txt), N tokens with the llama2.c tokenizer and no BOS, counted with the
# runtime's C++ SentencePiece tokenizer. Each line of output is one run: the runner's exit status, the prompt token
# count it reports, its first error line and the prefill chunk size the fixed runner logs. ADB_SERVER_SOCKET selects a
# remote adb server.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=$1 MODE=$2
D=/data/local/tmp/prefill-chunk/ci
adb shell "mkdir -p $D/bin $D/qnn $D/prompts"
adb push -q "$HERE/prompts/." "$D/prompts/" > /dev/null
adb push -q "$DIR/tokenizer.model" "$DIR/tokenizer.bin" "$D/" > /dev/null
adb shell "getprop ro.soc.model; getprop ro.product.model; getprop ro.build.version.release" | tr '\n' ' '
echo

run() {  # run <label> <command run on the device with $P set to the prompt>
  local label=$1 cmd=$2 n out
  for n in $SIZES; do
    out=$(adb shell "cd $D && P=\$(cat prompts/p$n.txt) && $cmd; echo EXIT=\$?" 2>&1)
    # Errors and the chunk-size log line; the tokenizer loader's fallback messages (hf_tokenizer, tiktoken) are not
    # errors.
    printf '%s n=%s exit=%s | prompt_tokens=%s | %s | %s\n' "$label" "$n" \
      "$(grep -o 'EXIT=[0-9]*' <<< "$out" | tail -1 | cut -d= -f2)" \
      "$(grep -oE 'Prompt Tokens: [0-9]+|"prompt_tokens":[0-9]+|total [0-9]+ prompt tokens|num_prompt_tokens [0-9]+' <<< "$out" | head -1 | grep -oE '[0-9]+$')" \
      "$(grep -v 'tokenizers:' <<< "$out" | grep -m1 -E 'Attempted to resize|Error resizing|Error|exceed|failed' | sed 's/.*\] //' | cut -c1-150)" \
      "$(grep -m1 -oE 'Prefill chunk size [0-9]+' <<< "$out")"
  done
}

case $MODE in
  xnnpack)
    adb push -q "$DIR/bin/llama_main" "$DIR/bin/llama_main_fixed" "$D/bin/" > /dev/null
    adb push -q "$DIR/stories110m_S128_C512_xnnpack.pte" "$D/" > /dev/null
    adb shell "chmod +x $D/bin/*; sha256sum $D/stories110m_S128_C512_xnnpack.pte $D/bin/llama_main $D/bin/llama_main_fixed"
    SIZES="127 128 129 257"
    for b in llama_main llama_main_fixed; do
      run "$b" "./bin/$b --model_path=stories110m_S128_C512_xnnpack.pte --tokenizer_path=tokenizer.model --prompt=\"\$P\" --num_bos=0 --max_new_tokens=4 --temperature=0 2>&1"
    done ;;
  qnn)
    adb push -q "$DIR/bin/qnn_llama_runner" "$D/bin/" > /dev/null
    adb push -q "$DIR/qnn/." "$D/qnn/" > /dev/null
    PTE=$(cd "$DIR" && find stories110m_qnn_SM8850 -maxdepth 1 -name '*.pte' | head -1)
    adb shell "mkdir -p $D/stories110m_qnn_SM8850"
    adb push -q "$DIR/$PTE" "$D/$PTE" > /dev/null
    adb shell "chmod +x $D/bin/*; sha256sum $D/$PTE $D/bin/qnn_llama_runner"
    SIZES="31 32 33 65 128 129 257"
    run qnn_llama_runner "LD_LIBRARY_PATH=$D/qnn ADSP_LIBRARY_PATH=$D/qnn ./bin/qnn_llama_runner --model_path=$PTE --tokenizer_path=tokenizer.bin --decoder_model_version=llama2 --eval_mode=1 --seq_len=511 --temperature=0 --prompt=\"\$P\" --output_path=out.txt 2>&1; cat out.txt" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
