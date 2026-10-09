# executorch-prefill-chunk

ExecuTorch's text runner prefills a prompt in chunks of `get_max_seq_len` tokens (`TextPrefiller`), but `export_llm`
(`extension/llm/export/builder.py`) has bounded the token input of KV-cache, dynamic-shape exports at
`max_seq_len - 1` since 1.1.0, while it publishes `get_max_seq_len = max_seq_len`. Every prompt of `max_seq_len`
tokens or more then fails: `Attempted to resize a bounded tensor with a maximum capacity of 127 elements to 128
elements` / `Error resizing tensor at input 0`. Through `generate()` this shows on exports with
`max_seq_length < max_context_length`; through `prefill()` it also shows on default exports (S == C).

`fix.patch` (against pytorch/executorch main `9875560`) has two parts: `builder.py` bounds the KV-cache token input at
`max_seq_len` again, and `create_text_llm_runner` sizes the prefill chunks by the method's real token bound
(`get_max_prefill_chunk_size`), so files already exported with the `- 1` bound work too.

## Host checks (`.github/workflows/executorch-prefill-chunk.yml`)

`prefill_check.py <file.pte> <tokenizer> [--expect observe|fixed]` records, for one export:

- what it publishes (`get_max_seq_len`, `get_max_context_len`) and what `forward` accepts (input 0's upper bound);
- `forward()` at input_pos 0 with S - 1, S and S + 1 tokens;
- a long prompt prefilled through `forward()` in chunks of the bound and of half of it (same next token);
- `TextLLMRunner.generate()` with prompts of exactly S - 1, S, S + 1 and 2S + 1 tokens (the binding exists from
  1.2.0), counted with the runner's own C++ tokenizer and checked against its `PyTorchObserver` `prompt_tokens`;
- `TextLLMRunner.prefill()` of exactly S tokens (it skips `generate()`'s context check, so it also covers S == C);
- the 2S + 1 prompt sent as `prefill()` pieces of at most the bound, then `generate("")`; the pieces are chosen so
  that their tokens concatenate to exactly the whole prompt's tokens.

With `--expect fixed` it exits 1 if any of these fails where it should work. `--expect observe` (the default) only
records.

`run.sh` modes (stock `export_llm`, KV cache, dynamic shape; `BACKEND=xnnpack` (default) or `mlx`):

- `repro <version> <model> <S> <C> <quant>`: install a release (1.0.1 to 1.5.1 with its torch, or `nightly`), export,
  check;
- `cross <export_version> <run_version> <model> <S> <C> <quant>`: export with one release, check with another;
- `fix <model> <S> <C> <quant>`: build main + `fix.patch` from source; check (a) the file exported by unpatched
  1.5.1 and (b) the file exported by the patched source, both with `--expect fixed`;
- `unit`: with that build, run the runner's C++ `test_runner` (including `PrefillChunkSizeTest`) and
  `test_builder.py -k dynamic_shape`.

Models: `stories110m` (Llama; the checkpoint ExecuTorch's own CI uses), `qwen3_0_6b`, `lfm2_350m` (hybrid
convolution and attention). Locally: `PYTHON=python3.12 ./run.sh repro 1.5.1 qwen3_0_6b 128 512 fp32`.

## Android (`.github/workflows/executorch-prefill-chunk-android.yml`, then `device.sh`)

`android.sh` (Linux) builds, from main `9875560`:

- `stories110m_S128_C512_xnnpack.pte`, the stock `export_llm` XNNPACK export, plus a host check of it;
- `bin/llama_main` (unpatched) and `bin/llama_main_fixed` (with `fix.patch`), Android arm64, XNNPACK;
- `bin/qnn_llama_runner` and a QNN export of the same checkpoint for SM8850 (`llama.py`, hybrid mode,
  `prefill_ar_len` 32, context 512), as a control: QNN exports use static shapes and their runner chunks by the
  program's own `ar_len`;
- `MANIFEST.txt` with the sha256 of every file.

`device.sh <unpacked artifact> xnnpack|qnn` runs them over adb with prompts of exactly the listed sizes
(`prompts/p<N>.txt`, " apple" repeated, counted with the C++ SentencePiece tokenizer, no BOS).

`android/PrefillAB.kt` does the same checks through the `executorch-android` AAR's `LlmModule` (`generate()` and the
`prefillPrompt()` workaround), run with `app_process` from a dex built against the AAR.
Build it with `kotlinc` against the AAR's `classes.jar`, convert with `d8` to `progs.dex` (including the AAR classes and
the Kotlin standard library), push the dex and the AAR's `jni/arm64-v8a/*.so`, then on the device:

```
CLASSPATH=<dir>/progs.dex LD_LIBRARY_PATH=<dir>/lib app_process -Djava.library.path=<dir>/lib /system/bin PrefillABKt \
  <model.pte> <tokenizer.model> <prompt dir> <bound> 127 128 129 257
```

The resize errors behind `LlmModule`'s `ExecuTorch Error 0x10` are in `logcat` (tag `ExecuTorch`).
