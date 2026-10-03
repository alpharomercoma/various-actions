"""Reproduce pytorch/executorch#23262 and check the fix, on one unpatched/patched export pair.

    python lfm2_state_check.py <unpatched.pte> <patched.pte> <tokenizer.json> [--json out.json]

Every check compares the two exports of the same checkpoint and recipe on this machine:
  leak        A new sequence at input_pos 0 on a reused module vs the same sequence on a freshly loaded module
              (4 random token pairs, final-token logits, threshold 1e-3). Unpatched should leak, patched should not.
  fresh       Final-token logits of a fresh sequence, patched vs unpatched. Must be identical.
  continue    A then B continued at positions len(A).. (multi-turn without a reset), patched vs unpatched. Identical.
  chunked     A fresh sequence prefilled in two chunks (positions 0.. and 3..), patched vs unpatched. Identical.
  reset       TextLLMRunner: generate A, reset(), generate B vs B on a fresh runner (greedy, 3 prompts).
  multiturn   TextLLMRunner: generate A, then B without reset, patched vs unpatched text. Identical.
Exit code 0 only if the unpatched export reproduces the leak and the patched export fixes it without changing
anything else.
"""
import argparse, hashlib, json, platform, random, sys

import torch
from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
from executorch.kernels import quantized  # noqa: F401
from executorch.extension.pybindings.portable_lib import _load_for_executorch

try:
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
except RuntimeError as e:  # the Windows wheel ships without the LLM runner bindings
    TextLLMRunner = None
    RUNNER_MISSING = str(e)

ap = argparse.ArgumentParser()
ap.add_argument("unpatched"); ap.add_argument("patched"); ap.add_argument("tokenizer")
ap.add_argument("--json")
a = ap.parse_args()
PTE = {"unpatched": a.unpatched, "patched": a.patched}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fwd(m, toks, start):
    return m.forward((torch.tensor([toks], dtype=torch.long), torch.tensor([start], dtype=torch.long)))[0].reshape(-1).float()


def step(m, toks, start=0):
    out = None
    for i, t in enumerate(toks):
        out = fwd(m, [t], start + i)
    return out


maxdiff = lambda x, y: float((x - y).abs().max())
rng = random.Random(0)
pairs = [([1] + [rng.randrange(100, 8000) for _ in range(la)], [1] + [rng.randrange(100, 8000) for _ in range(lb)])
         for la, lb in [(1, 4), (6, 5), (12, 8), (3, 3)]]
res = {"platform": f"{platform.system()} {platform.machine()}", "python": platform.python_version(),
       "torch": torch.__version__, "sha256": {k: sha(v) for k, v in PTE.items()}}
try:
    import executorch.version as ev
    res["executorch"] = f"{ev.__version__} ({ev.git_version[:8]})"
except Exception:  # noqa: BLE001
    res["executorch"] = "unknown"

fresh, cont, chunk = {}, {}, {}
nonfinite = 0
for side, pte in PTE.items():
    leaks, worst = 0, 0.0
    for k, (A, B) in enumerate(pairs):
        f = step(_load_for_executorch(pte), B)
        m = _load_for_executorch(pte); step(m, A); after = step(m, B)
        nonfinite += int(not (torch.isfinite(f).all() and torch.isfinite(after).all()))
        d = maxdiff(f, after); leaks += d > 1e-3; worst = max(worst, d)
        fresh[(side, k)] = f
        m = _load_for_executorch(pte); step(m, A); cont[(side, k)] = step(m, B[1:], start=len(A))
        m = _load_for_executorch(pte); fwd(m, B[:3], 0); chunk[(side, k)] = fwd(m, B[3:], 3)
    res[f"leak_{side}"] = {"pairs_leaking": leaks, "of": len(pairs), "max_abs_diff": worst}
res["fresh_max_abs_diff"] = max(maxdiff(fresh[("unpatched", k)], fresh[("patched", k)]) for k in range(len(pairs)))
res["continue_max_abs_diff"] = max(maxdiff(cont[("unpatched", k)], cont[("patched", k)]) for k in range(len(pairs)))
res["chunked_max_abs_diff"] = max(maxdiff(chunk[("unpatched", k)], chunk[("patched", k)]) for k in range(len(pairs)))

chat = lambda q: f"<|startoftext|><|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
turn = lambda q: f"<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
A = "Write a short poem about the ocean at night."
Bs = ["Tell me a short story about a dragon who is afraid of fire.",
      "Explain how vaccines train the immune system.", "List five uses for a paperclip."]


def gen(r, p, n=48):
    out = []
    r.generate(p, GenerationConfig(echo=False, max_new_tokens=n, temperature=0.0), out.append)
    return "".join(out)


if TextLLMRunner is not None:
    multiturn = {}
    for side, pte in PTE.items():
        same = 0
        for i, B in enumerate(Bs):
            f = gen(TextLLMRunner(pte, a.tokenizer), chat(B))
            r = TextLLMRunner(pte, a.tokenizer); gen(r, chat(A)); r.reset()
            same += gen(r, chat(B)) == f
            r = TextLLMRunner(pte, a.tokenizer); gen(r, chat(A)); multiturn[(side, i)] = gen(r, turn(B))
        res[f"reset_{side}"] = {"same_as_fresh": same, "of": len(Bs)}
    res["multiturn_identical"] = sum(multiturn[("unpatched", i)] == multiturn[("patched", i)] for i in range(len(Bs)))

runner = TextLLMRunner is not None
res["runner_checks"] = "ran" if runner else f"skipped: {RUNNER_MISSING}"
res["nonfinite_logit_vectors"] = nonfinite
checks = {
    "all compared logits are finite": nonfinite == 0,
    "TextLLMRunner checks ran (only the Windows wheel may lack them)": runner or platform.system() == "Windows",
    "unpatched export leaks (reproduces #23262)": res["leak_unpatched"]["pairs_leaking"] >= 3
    and (not runner or res["reset_unpatched"]["same_as_fresh"] < len(Bs)),
    "patched export does not leak": res["leak_patched"]["pairs_leaking"] == 0
    and (not runner or res["reset_patched"]["same_as_fresh"] == len(Bs)),
    "fresh sequence unchanged by the fix": res["fresh_max_abs_diff"] == 0.0,
    "multi-turn continuation unchanged by the fix": res["continue_max_abs_diff"] == 0.0
    and (not runner or res["multiturn_identical"] == len(Bs)),
    "chunked prefill unchanged by the fix": res["chunked_max_abs_diff"] == 0.0,
}
res["checks"] = checks
lu, lp = res["leak_unpatched"], res["leak_patched"]
print(f"platform {res['platform']} | python {res['python']} | torch {res['torch']} | executorch {res['executorch']}")
for side in PTE:
    print(f"sha256 {side:9s} {res['sha256'][side]}")
print(f"leak     unpatched {lu['pairs_leaking']}/{lu['of']} (max {lu['max_abs_diff']:.4f}) | "
      f"patched {lp['pairs_leaking']}/{lp['of']} (max {lp['max_abs_diff']:.4f})")
if runner:
    ru, rp = res["reset_unpatched"], res["reset_patched"]
    print(f"reset    after reset() == fresh: unpatched {ru['same_as_fresh']}/{ru['of']} | patched {rp['same_as_fresh']}/{rp['of']}")
else:
    print(f"reset    TextLLMRunner checks {res['runner_checks']}")
print(f"unchanged by fix: fresh {res['fresh_max_abs_diff']} | continue {res['continue_max_abs_diff']} | "
      f"chunked {res['chunked_max_abs_diff']}" + (f" | multi-turn text identical {res['multiturn_identical']}/{len(Bs)}" if runner else ""))
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
if a.json:
    with open(a.json, "w") as f:
        json.dump(res, f, indent=2)
sys.exit(0 if all(checks.values()) else 1)
