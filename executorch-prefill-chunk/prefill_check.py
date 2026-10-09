"""Measure the prefill chunk contract of an exported ExecuTorch LLM.

The text runner prefills a long prompt in chunks of `get_max_seq_len` tokens (TextPrefiller). This script reads what
the program advertises (`get_max_seq_len`, `get_max_context_len`) and what its `forward` actually accepts (the upper
bound of input 0's token dimension), then observes, on the same file:

  module   forward() at input_pos 0 with bound, bound + 1 and advertised (S - 1, S, S + 1) tokens;
  chunked  a long prompt prefilled through forward() in chunks of the real bound, and in chunks of half of it:
           both must succeed and give the same next token (chunk size must not change the result);
  runner   TextLLMRunner.generate() with prompts of exactly S - 1, S, S + 1 and 2S + 1 tokens (if the binding exists);
  workaround  the same 2S + 1 prompt sent as TextLLMRunner.prefill() pieces of at most `bound` tokens, then generate("").

  python prefill_check.py <file.pte> <tokenizer> [--json out.json]

Every observation is recorded with the error text the runtime printed, so a pass/fail table can be built from the
JSON alone. Nothing here depends on how the file was quantized.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import sys
import tempfile


@contextlib.contextmanager
def captured_stderr():
    """Capture what the C++ runtime writes to fd 2 (ET_LOG) and fd 1 (the runner's PyTorchObserver stats) during a
    call. `text` is fd 2, `stdout` is fd 1."""
    buf = {"text": "", "stdout": ""}
    sys.stdout.flush()
    sys.stderr.flush()
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    with tempfile.TemporaryFile(mode="w+b") as out, tempfile.TemporaryFile(mode="w+b") as err:
        os.dup2(out.fileno(), 1)
        os.dup2(err.fileno(), 2)
        try:
            yield buf
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, keep in saved.items():
                os.dup2(keep, fd)
                os.close(keep)
            err.seek(0)
            out.seek(0)
            buf["text"] = err.read().decode("utf-8", "replace")
            buf["stdout"] = out.read().decode("utf-8", "replace")


def observed_prompt_tokens(text: str) -> int | None:
    """prompt_tokens from the runner's PyTorchObserver line: the count the runner itself encoded."""
    for line in text.splitlines():
        if "PyTorchObserver" in line:
            with contextlib.suppress(Exception):
                return int(json.loads(line.split("PyTorchObserver", 1)[1])["prompt_tokens"])
    return None


def error_lines(text: str) -> list[str]:
    keep = ("Error", "error", "exceed", "Check failed", "resiz", "Invalid", "failed")
    return [line.strip()[:300] for line in text.splitlines() if any(k in line for k in keep)][:6]


def load_kernels() -> None:
    # The LLM custom ops and quantized kernels register themselves on import; exports need them.
    for mod in ("executorch.extension.llm.custom_ops.custom_ops", "executorch.kernels.quantized"):
        with contextlib.suppress(Exception):
            __import__(mod)


def program_facts(path: str) -> dict:
    from executorch.runtime import Runtime

    rt = Runtime.get()
    prog = rt.load_program(path)
    names = set(prog.method_names)

    def const(name: str):
        if name not in names:
            return None
        return prog.load_method(name).execute([])[0]

    meta = prog.metadata("forward")
    sizes = list(meta.input_tensor_meta(0).sizes())  # a dynamic dimension reports its upper bound
    return {
        "max_seq_len": const("get_max_seq_len"),
        "max_context_len": const("get_max_context_len"),
        "enable_dynamic_shape": const("enable_dynamic_shape"),
        "use_kv_cache": const("use_kv_cache"),
        "input0_sizes": sizes,
        "bound": sizes[1] if len(sizes) >= 2 else None,
    }


def forward_call(path: str, n_tokens: int) -> dict:
    """One forward() of n tokens at input_pos 0, on a freshly loaded method."""
    import torch
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(path).load_method("forward")
    tokens = torch.full((1, n_tokens), 1, dtype=torch.long)
    with captured_stderr() as err:
        try:
            method.execute([tokens, torch.tensor([0], dtype=torch.long)])
            ok = True
        except Exception as e:  # noqa: BLE001 - the error is the observation
            ok, exc = False, f"{type(e).__name__}: {e}"
    out = {"n": n_tokens, "ok": ok, "stderr": error_lines(err["text"])}
    if not ok:
        out["exception"] = exc[:300]
    return out


def chunked_prefill(path: str, tokens: list[int], chunk: int) -> tuple[bool, int | None, list[str]]:
    """Prefill `tokens` through forward() in chunks of `chunk`; return (ok, next token, errors)."""
    import torch
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(path).load_method("forward")
    pos, logits = 0, None
    with captured_stderr() as err:
        try:
            while pos < len(tokens):
                piece = tokens[pos : pos + chunk]
                out = method.execute([torch.tensor([piece], dtype=torch.long), torch.tensor([pos], dtype=torch.long)])
                logits = out[0]
                pos += len(piece)
            ok = True
        except Exception:  # noqa: BLE001
            ok = False
    nxt = int(logits.reshape(-1, logits.shape[-1])[-1].argmax()) if (ok and logits is not None) else None
    return ok, nxt, error_lines(err["text"])


class Prompts:
    """Prompts whose token count, as the runner's own C++ tokenizer encodes them, is exactly a target.

    The Python HuggingFaceTokenizer adds the template's BOS even with bos=0, the C++ HFTokenizer the runner loads does
    not, so counting with the Python one is off by one for LFM2. Load the C++ classes in the order
    llm_runner_helper.cpp's load_tokenizer tries them (HF json, then SentencePiece, then llama2c), and fall back to
    the Python tokenizer only where the C++ bindings are absent (recorded in `impl`).
    """

    def __init__(self, tokenizer_path: str):
        self.impl, self.tok = None, None
        with contextlib.suppress(Exception):
            import pytorch_tokenizers as pt

            for name in ("CppHFTokenizer", "CppSPTokenizer", "CppLlama2cTokenizer"):
                cls = getattr(pt, name, None)
                if cls is None:
                    continue
                tok = cls()
                with contextlib.suppress(Exception):
                    if str(tok.load(tokenizer_path)).endswith("Ok") or tok.vocab_size() > 0:
                        self.impl, self.tok = name, tok
                        break
        if self.tok is None:
            from pytorch_tokenizers import get_tokenizer

            self.impl, self.tok = (
                "python " + type(get_tokenizer(tokenizer_path)).__name__,
                get_tokenizer(tokenizer_path),
            )

    def tokens(self, text: str) -> list[int]:
        if self.impl.startswith("Cpp"):
            return list(self.tok.encode(text, 0, 0))
        return list(self.tok.encode(text, bos=0, eos=0))

    def count(self, text: str) -> int:
        return len(self.tokens(text))

    def exactly(self, n: int, lead: str = "") -> str:
        words = ["apple"] * max(1, n)
        text = lead + " ".join(words)
        # Most tokenizers give one token per " apple"; walk to the exact count either way.
        for _ in range(4 * n + 50):
            c = self.count(text)
            if c == n:
                return text
            words = words[:-1] if c > n else words + ["apple"]
            text = lead + " ".join(words)
        raise RuntimeError(f"could not build a prompt of exactly {n} tokens")

    def split(self, n: int, bound: int) -> tuple[str, list[str]] | None:
        """A prompt of n tokens and pieces of at most `bound` tokens whose separate encodings concatenate to exactly the
        whole prompt's tokens (so prefill() of the pieces and generate() of the whole see the same token sequence)."""
        sizes, left = [], n
        while left > 0:
            sizes.append(min(bound, left))
            left -= sizes[-1]
        for lead in ("", " "):
            pieces = [self.exactly(sizes[0])] + [self.exactly(k, lead) for k in sizes[1:]]
            whole = (" " if lead == "" else "").join(pieces)
            if self.tokens(whole) == [t for p in pieces for t in self.tokens(p)]:
                return whole, pieces
        return None


def runner_cases(path: str, tok_path: str, facts: dict) -> dict | None:
    try:
        from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    except Exception as e:  # noqa: BLE001 - absent before ExecuTorch 1.2
        return {"available": False, "reason": f"{type(e).__name__}: {e}"[:200]}
    s, ctx, bound = facts["max_seq_len"], facts["max_context_len"], facts["bound"]
    prompts = Prompts(tok_path)

    def config():
        cfg = GenerationConfig()
        for k, v in (("echo", False), ("max_new_tokens", 4), ("temperature", 0.0), ("num_bos", 0), ("num_eos", 0)):
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    targets = [s - 1, s, s + 1]
    long_split = None
    if 2 * s + 1 + 8 < ctx:
        targets.append(2 * s + 1)
        if bound:
            long_split = prompts.split(2 * s + 1, bound)
    cases = []
    for n in targets:
        runner = TextLLMRunner(path, tok_path)
        text = long_split[0] if (long_split and n == 2 * s + 1) else prompts.exactly(n)
        pieces = []
        with captured_stderr() as err:
            try:
                runner.generate(text, config(), pieces.append)
                ok = True
            except Exception as e:  # noqa: BLE001
                ok, exc = False, f"{type(e).__name__}: {e}"
        row = {
            "n": n,
            "ok": ok,
            "runner_prompt_tokens": observed_prompt_tokens(err["stdout"] + err["text"]),
            "pieces": len(pieces),
            "text": "".join(pieces),
            "stderr": error_lines(err["text"]),
        }
        if not ok:
            row["exception"] = exc[:200]
        cases.append(row)

    workaround = None
    if long_split and hasattr(TextLLMRunner, "prefill"):
        # Same 2S + 1 prompt, sent in pieces of at most `bound` tokens, then generation from the prefilled state.
        n = 2 * s + 1
        runner = TextLLMRunner(path, tok_path)
        sizes = [prompts.count(p) for p in long_split[1]]
        out = []
        with captured_stderr() as err:
            try:
                for piece in long_split[1]:
                    runner.prefill(piece, config())
                runner.generate("", config(), out.append)
                ok = True
            except Exception as e:  # noqa: BLE001
                ok, exc = False, f"{type(e).__name__}: {e}"
        workaround = {
            "n": n,
            "piece_sizes": sizes,
            "ok": ok,
            "pieces": len(out),
            "text": "".join(out),
            "stderr": error_lines(err["text"]),
        }
        if not ok:
            workaround["exception"] = exc[:200]
    long_case = next((c for c in cases if c["n"] == 2 * s + 1), None)
    same = None
    if workaround and long_case and workaround["ok"] and long_case["ok"]:
        # With a runner that chunks at the real bound, generate(2S + 1 tokens) and the bound-sized prefill pieces see
        # the same chunks, so their greedy continuations must match.
        same = long_case["text"] == workaround["text"]
    return {
        "available": True,
        "tokenizer": prompts.impl,
        "cases": cases,
        "workaround": workaround,
        "long_matches_workaround": same,
        "long_prompt_splits_exactly": long_split is not None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pte")
    ap.add_argument("tokenizer")
    ap.add_argument("--json")
    args = ap.parse_args()
    load_kernels()

    import executorch
    import torch

    try:
        from executorch import version as etv

        et_version = f"{etv.__version__} ({etv.git_version[:12]})"
    except Exception:  # noqa: BLE001
        et_version = getattr(executorch, "__version__", "unknown")
    with open(args.pte, "rb") as f:
        data = f.read()
    facts = program_facts(args.pte)
    s, bound, ctx = facts["max_seq_len"], facts["bound"], facts["max_context_len"]
    result = {
        "file": os.path.basename(args.pte),
        "sha256": hashlib.sha256(data).hexdigest(),
        "platform": f"{platform.system()} {platform.machine()}",
        "executorch": et_version,
        "torch": torch.__version__,
        "facts": facts,
    }

    sizes = sorted({n for n in (bound, (bound or 0) + 1, s - 1, s, s + 1) if n and n > 0})
    result["module"] = [forward_call(args.pte, n) for n in sizes]

    # A prompt longer than two chunks plus a partial one, prefilled at the real bound and at half of it.
    n_long = min(2 * (bound or s) + 7, ctx - 8)
    tokens = [(7 * i) % 200 + 10 for i in range(n_long)]
    ok_a, next_a, err_a = chunked_prefill(args.pte, tokens, bound or s)
    half = max(1, (bound or s) // 2)
    ok_b, next_b, err_b = chunked_prefill(args.pte, tokens, half)
    result["chunked"] = {
        "n": n_long,
        "at_bound": {"chunk": bound, "ok": ok_a, "next_token": next_a, "stderr": err_a},
        "at_half": {"chunk": half, "ok": ok_b, "next_token": next_b, "stderr": err_b},
        "same_next_token": ok_a and ok_b and next_a == next_b,
    }
    result["runner"] = runner_cases(args.pte, args.tokenizer, facts)

    mismatch = s is not None and bound is not None and bound < s
    result["verdict"] = {
        "advertised_minus_bound": None if (s is None or bound is None) else s - bound,
        "mismatch": mismatch,
        "chunk_eq_window": s == ctx,
    }
    lines = [
        f"platform {result['platform']} | executorch {et_version} | torch {torch.__version__}",
        f"file {result['file']} sha256 {result['sha256'][:16]}",
        f"advertised get_max_seq_len={s} get_max_context_len={ctx} | forward input0 {facts['input0_sizes']} (bound {bound})",
        "module " + " ".join(f"{r['n']}:{'ok' if r['ok'] else 'FAIL'}" for r in result["module"]),
        (
            f"chunked n={n_long} at_bound({bound}):{'ok' if ok_a else 'FAIL'} at_half({half}):{'ok' if ok_b else 'FAIL'} "
            f"same_next_token={result['chunked']['same_next_token']}"
        ),
    ]
    r = result["runner"]
    if r and r.get("available"):
        lines.append(
            f"runner ({r.get('tokenizer')}) "
            + " ".join(
                f"{c['n']}:{'ok' if c['ok'] else 'FAIL'}"
                + (
                    f"[runner saw {c['runner_prompt_tokens']}]"
                    if c.get("runner_prompt_tokens") not in (None, c["n"])
                    else ""
                )
                for c in r["cases"]
            )
        )
        if r.get("workaround"):
            w = r["workaround"]
            lines.append(
                f"workaround n={w['n']} pieces {w['piece_sizes']}: {'ok' if w['ok'] else 'FAIL'}"
                f" | runner output == workaround output: {r.get('long_matches_workaround')}"
            )
        first = next((c for c in r["cases"] if not c["ok"]), None)
        if first:
            lines.append(f"first runner failure ({first['n']} tokens): {' | '.join(first['stderr'][:2])}")
    else:
        lines.append(f"runner unavailable: {(r or {}).get('reason')}")
    print("\n".join(lines))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
