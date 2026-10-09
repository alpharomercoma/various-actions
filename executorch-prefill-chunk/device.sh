#!/usr/bin/env bash
# Run the on-device checks with the files from the executorch-prefill-chunk-android workflow (its `prefill-android`
# artifact) or, for vulkan, the executorch-prefill-chunk-vulkan workflow (`prefill-vulkan`), unpacked into <dir>, over adb:
#
#   device.sh <dir> xnnpack   llama_main and llama_main_fixed on the stock export_llm XNNPACK export (S=128, C=512):
#                             prompts of exactly 127, 128, 129 and 257 tokens
#   device.sh <dir> vulkan    llama_main_vk and llama_main_vk_fixed (Vulkan backend) on stock export_llm Vulkan exports
#                             (fp32 and 8da4w, S=128, C=512): prompts of exactly 127, 128, 129 and 257 tokens
#   device.sh <dir> mediatek  mtk_llama_executor_runner (NeuroPilot) on the examples/mediatek Qwen2.5-0.5B export for
#                             MT6991 (128-token prompt batch): prompts of 126 to 257 tokens
#   device.sh <dir> qnn [SOC] qnn_llama_runner on the QNN export for SOC (SM8850 by default, or SM8750; prefill_ar_len
#                             32, context 512):
#                             prompts of exactly 31, 32, 33, 65, 128, 129 and 257 tokens
#
# Prompts are " apple" repeated (prompts/p<N>.txt), N tokens with the llama2.c tokenizer and no BOS, counted with the
# runtime's C++ SentencePiece tokenizer. Each line of output is one run: the runner's exit status, the prompt token
# count it reports (qnn_llama_runner adds a BOS token), its first error line and the prefill chunking it logs. ADB_SERVER_SOCKET selects a
# remote adb server.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=$1 MODE=$2
D=/data/local/tmp/prefill-chunk/ci
adb shell "mkdir -p $D/bin $D/qnn $D/prompts"
adb push -q "$HERE/prompts/." "$D/prompts/" > /dev/null
for f in tokenizer.model tokenizer.bin; do [ -f "$DIR/$f" ] && adb push -q "$DIR/$f" "$D/" > /dev/null; done
adb shell "getprop ro.soc.model; getprop ro.product.model; getprop ro.build.version.release" | tr '\n' ' '
echo

run() {  # run <label> <command run on the device with $P set to the prompt>
  local label=$1 cmd=$2 n out
  for n in $SIZES; do
    out=$(adb shell "cd $D && P=\$(cat ${PROMPTS:-prompts}/p$n.txt) && $cmd; echo EXIT=\$?" 2>&1)
    # Errors and the chunk-size log line; the tokenizer loader's fallback messages (hf_tokenizer, tiktoken, sentencepiece) are not
    # errors.
    printf '%s n=%s exit=%s | prompt_tokens=%s generated=%s | %s | %s\n' "$label" "$n" \
      "$(grep -o 'EXIT=[0-9]*' <<< "$out" | tail -1 | cut -d= -f2)" \
      "$(grep -oE 'Prompt Tokens: [0-9]+|"prompt_tokens":[0-9]+|total [0-9]+ prompt tokens|num_prompt_tokens [0-9]+' <<< "$out" | head -1 | grep -oE '[0-9]+')" \
      "$(grep -oE '"generated_tokens":[0-9]+' <<< "$out" | head -1 | grep -oE '[0-9]+')" \
      "$(grep -v -E 'tokenizers:|^Error message:|load tokenizer|ModelProto|tokenizer artifact' <<< "$out" | grep -m1 -E 'Attempted to resize|Error resizing|Error|exceed|failed' | sed 's/.*\] //' | cut -c1-150)" \
      "$(grep -m1 -oE 'Prefill chunk size [0-9]+|AR-[0-9]+ \* [0-9]+ iters' <<< "$out")"
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
    SOC=${3:-SM8850}  # device.sh <dir> qnn [SM8850|SM8750]
    PTE=$(cd "$DIR" && find "stories110m_qnn_$SOC" -maxdepth 1 -name '*.pte' | head -1)
    adb shell "mkdir -p $D/stories110m_qnn_$SOC"
    adb push -q "$DIR/$PTE" "$D/$PTE" > /dev/null
    adb shell "chmod +x $D/bin/*; sha256sum $D/$PTE $D/bin/qnn_llama_runner"
    SIZES="31 32 33 65 128 129 257"
    # qnn_llama_runner collects prompts from "--prompt <text>" pairs only (CollectPrompts); "--prompt=<text>" runs nothing.
    run qnn_llama_runner "rm -f out.txt; LD_LIBRARY_PATH=$D/qnn ADSP_LIBRARY_PATH=$D/qnn ./bin/qnn_llama_runner --model_path=$PTE --tokenizer_path=tokenizer.bin --decoder_model_version=llama2 --eval_mode=1 --seq_len=511 --temperature=0 --output_path=out.txt --prompt \"\$P\" 2>&1; rc=\$?; cat out.txt; (exit \$rc)" ;;
  vulkan)
    # From the executorch-prefill-chunk-vulkan workflow's artifact: llama_main with the Vulkan backend, main without
    # and with fix.patch, on the stock export_llm Vulkan exports (fp32, and 8da4w with force_fp16), S=128 C=512.
    adb push -q "$DIR/bin/llama_main_vk" "$DIR/bin/llama_main_vk_fixed" "$D/bin/" > /dev/null
    # SKIP_MODEL_PUSH=1 when the files are already on the device (the sha256sum below shows which ones are there).
    [ -n "${SKIP_MODEL_PUSH:-}" ] || adb push -q "$DIR/stories110m_S128_C512_vulkan.pte" "$DIR/stories110m_S128_C512_vulkan_8da4w.pte" "$D/" > /dev/null
    adb shell "chmod +x $D/bin/*; sha256sum $D/stories110m_S128_C512_vulkan*.pte $D/bin/llama_main_vk*"
    SIZES="127 128 129 257"
    for f in stories110m_S128_C512_vulkan stories110m_S128_C512_vulkan_8da4w; do
      for b in llama_main_vk llama_main_vk_fixed; do
        run "$b $f" "./bin/$b --model_path=$f.pte --tokenizer_path=tokenizer.model --prompt=\"\$P\" --num_bos=0 --max_new_tokens=4 --temperature=0 2>&1"
      done
    done ;;
  models)
    # From phone_models.sh's output (qwen3_0_6b and lfm2_350m, XNNPACK fp32 and Vulkan 8da4w fp16, S=128, C=512) with
    # bin/llama_main, bin/llama_main_fixed, bin/llama_main_vk and bin/llama_main_vk_fixed copied in. Prompts are each
    # model's own (prompts_<model>/p<N>.txt), exactly N tokens with its tokenizer, no BOS.
    adb push -q "$DIR/bin/." "$D/bin/" > /dev/null
    adb shell "chmod +x $D/bin/*"
    SIZES="127 128 129 257"
    for m in ${MODELS:-qwen3_0_6b lfm2_350m}; do
      adb shell "mkdir -p $D/prompts_$m"
      adb push -q "$DIR/prompts_$m/." "$D/prompts_$m/" > /dev/null
      adb push -q "$DIR/$m.tokenizer.json" "$D/" > /dev/null
      # KINDS selects the files (default both); each is pushed on its own, with retries, for slow or flaky links.
      for k in ${KINDS:-xnnpack vulkan_8da4w}; do
        for i in 1 2 3 4; do adb push -q "$DIR/${m}_S128_C512_$k.pte" "$D/" > /dev/null && break; done
      done
      adb shell "sha256sum $D/${m}_S128_C512_*.pte"
      for pair in "xnnpack:llama_main" "xnnpack:llama_main_fixed" "vulkan_8da4w:llama_main_vk" "vulkan_8da4w:llama_main_vk_fixed"; do
        case " ${KINDS:-xnnpack vulkan_8da4w} " in *" ${pair%%:*} "*) ;; *) continue ;; esac
        f=${m}_S128_C512_${pair%%:*} b=${pair#*:}
        PROMPTS=prompts_$m run "$b $f" "./bin/$b --model_path=$f.pte --tokenizer_path=$m.tokenizer.json --prompt=\"\$P\" --num_bos=0 --max_new_tokens=4 --temperature=0 2>&1"
      done
    done ;;
  mediatek)
    # From the executorch-prefill-chunk-mediatek workflow's artifact: mtk_llama_executor_runner (NeuroPilot, MT6991),
    # main without and with fix.patch, on the examples/mediatek Qwen2.5-0.5B-Instruct export (128-token prompt batch,
    # cache 512). Prompts are prompts/q<N>.txt from the artifact; the runner adds its BOS setting itself.
    M=$D/mtk
    adb shell "mkdir -p $M/lib $M/prompts"
    adb push -q "$DIR/lib/." "$M/lib/" > /dev/null
    adb push -q "$DIR/bin/mtk_llama_executor_runner" "$DIR/bin/mtk_llama_executor_runner_fixed" "$M/" > /dev/null
    adb push -q "$DIR/prompts/." "$M/prompts/" > /dev/null
    adb push -q "$DIR/tokenizer.json" "$DIR"/embedding_*_fp32.bin "$M/" > /dev/null
    PTE=$(cd "$DIR" && find . -name '*.pte' | head -1 | sed 's|^\./||')
    EMB=$(cd "$DIR" && ls embedding_*_fp32.bin | head -1)
    adb shell "mkdir -p $M/$(dirname "$PTE")"
    adb push -q "$DIR/$PTE" "$M/$PTE" > /dev/null
    adb shell "chmod +x $M/mtk_llama_executor_runner*; sha256sum $M/$PTE $M/mtk_llama_executor_runner*"
    for b in mtk_llama_executor_runner mtk_llama_executor_runner_fixed; do
      for n in 126 127 128 129 255 256 257; do
        out=$(adb shell "cd $M && LD_LIBRARY_PATH=$M/lib:\$LD_LIBRARY_PATH ./$b --max_response=4 \
          --prompt_token_batch_size=128 --cache_size=512 --hidden_size=896 --num_head=14 --num_layer=24 \
          --max_token_length=32768 --rot_emb_base=1000000 --input_type=fp32 --output_type=fp32 --cache_type=fp32 \
          --mask_type=fp32 --rot_emb_type=fp32 --vocab_size=151936 --bos_token=151643 --eos_token=151645 \
          --tokenizer_type=hf --tokenizer_path=tokenizer.json --token_embedding_path=$EMB \
          --model_package_paths=$PTE --prompt_file=prompts/q$n.txt 2>&1; echo EXIT=\$?" 2>&1)
        printf '%s n=%s exit=%s | prompt_tokens=%s | %s | response=%s\n' "$b" "$n" \
          "$(grep -o 'EXIT=[0-9]*' <<< "$out" | tail -1 | cut -d= -f2)" \
          "$(awk '/\[Input Prompt Tokens\]/{getline; print}' <<< "$out" | tr -cs '0-9' '\n' | grep -c .)" \
          "$(grep -v 'tokenizers:' <<< "$out" | grep -m1 -E 'rror|fail|Abort' | sed 's/.*\] //' | cut -c1-150)" \
          "$(awk '/\[Real-time Response\]/{getline; print; exit}' <<< "$out" | cut -c1-40)"
      done
    done ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
