# executorch-lfm2-conv-state

[pytorch/executorch#23262](https://github.com/pytorch/executorch/issues/23262): in exported LFM2 and LFM2.5 models,
`ShortConv.conv_state` keeps the previous sequence's last two convolution inputs, so a new sequence after
`TextLLMRunner.reset()` / `LlmModule.resetContext()` differs from the same sequence on a freshly loaded model.
`fix.patch` (against pytorch/executorch main) clears the state when a sequence starts at position 0.

The workflow runs on `ubuntu-24.04` (x86_64), `ubuntu-24.04-arm` (aarch64), `macos-15` (arm64) and `windows-2025`
(x86_64), with the executorch nightly built from main (`1.6.0.dev20261002`, git `0f5dc8e8`) and the 1.5.1 release:

- `run.sh unit <runtime>`: the four regression tests from the patch must fail on unpatched code and pass with the fix;
  on nightly, `test_qwen3_5_attention.py` and `test_transformer_block.py` from the same commit must still pass.
- `run.sh e2e <runtime> <model>`: export the model twice with the README recipe (`lfm2_xnnpack_q8da4w.yaml`), once
  unpatched and once with the fix, then `lfm2_state_check.py` requires that the unpatched export leaks, the patched
  export does not, and that fresh sequences, multi-turn continuation and chunked prefill are bit-identical between
  the two.

Windows: the `win_amd64` wheels (1.5.1 and nightly) export fine but their runtime registers neither the
`llama::custom_sdpa` kernel nor the `TextLLMRunner` bindings. Windows jobs therefore export with the same recipe plus
`model.use_sdpa_with_kv_cache=False` (recipe `no_custom_sdpa`; the conv layers are unchanged) and skip the runner checks.
One Linux job runs the same recipe as a control.

The fix is applied by copying the patched `short_conv.py` over the installed wheel's copy (the only file the patch
changes outside tests). Tests run from a directory without the executorch sources so they import the installed package.

Run locally: `PYTHON=/path/to/venv/python ./run.sh e2e nightly lfm2_5_350m`.
