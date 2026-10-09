# executorch-prefill-chunk

ExecuTorch's text runner prefills a prompt longer than `get_max_seq_len` in chunks of `get_max_seq_len` tokens
(`TextPrefiller`), but `export_llm` (`extension/llm/export/builder.py`) bounds the token input of a KV-cache export at
`max_seq_len - 1` while it publishes `get_max_seq_len = max_seq_len`. On an export whose prefill chunk is smaller than
its context (`max_seq_length < max_context_length`), every prompt of `max_seq_len` tokens or more fails:
`Attempted to resize a bounded tensor with a maximum capacity of 127 elements to 128 elements` /
`Error resizing tensor at input 0`. `fix.patch` (against pytorch/executorch main) sizes the runner's prefill chunks
by the token input's real bound.

`prefill_check.py <file.pte> <tokenizer>` records, for one export:

- what it advertises (`get_max_seq_len`, `get_max_context_len`) and what `forward` accepts (input 0's upper bound);
- `forward()` at input_pos 0 with the bound, bound + 1 and S - 1, S, S + 1 tokens;
- a prompt of two chunks plus a partial one prefilled through `forward()` at the bound and at half of it (both must
  give the same next token);
- `TextLLMRunner.generate()` with prompts of exactly S - 1, S, S + 1 and 2S + 1 tokens (binding exists from 1.2.0);
- the 2S + 1 prompt sent as `prefill()` pieces of at most the bound, then `generate("")`, an app-side workaround;
  with the fix, `generate()` of the whole prompt must produce the same text.

`run.sh` modes (all exports use stock `export_llm`, KV cache, dynamic shape, XNNPACK):

- `repro <version> <model> <S> <C> <quant>`: install a release (1.0.1 to 1.5.1 with its torch, or the nightly), export,
  check;
- `cross <export_version> <run_version> ...`: export with one release, check with another;
- `fix <model> <S> <C> <quant>`: build main + `fix.patch` from source, export, check;
- `unit`: with that build, run the runner's C++ `test_runner`, including the new `PrefillChunkSizeTest`.

Models: `stories110m` (Llama architecture; the checkpoint ExecuTorch's own CI uses), `qwen3_0_6b`, `lfm2_350m`
(hybrid convolution and attention). Run locally: `PYTHON=python3.12 ./run.sh repro 1.5.1 qwen3_0_6b 128 512 fp32`.
