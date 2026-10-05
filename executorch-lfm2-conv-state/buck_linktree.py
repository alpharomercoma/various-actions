"""Approximate a Buck python_unittest link tree from the repo's BUCK files.

    python buck_linktree.py <executorch-checkout> <package/dir:target> <out-dir>

Evaluates BUCK files with stub rule functions, walks the dependency closure of the target through
//executorch/... python targets, and copies each target's srcs to <out-dir>/<base_module path>/. External deps
(//caffe2:torch, fbsource//third-party/pypi/..., fbcode//pytorch/...) are expected from the interpreter's
site-packages. The out-dir is then the only place `executorch` can be imported from, like a Buck link tree.
"""
import glob as globmod
import os
import shutil
import sys

ROOT, START, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
targets = {}


class Rule:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attr):
        return Rule(f"{self.name}.{attr}")

    def __call__(self, *a, **k):
        return None


def load_buck(pkg):
    path = os.path.join(ROOT, pkg, "BUCK")
    if pkg in loaded:
        return
    loaded.add(pkg)
    if not os.path.exists(path):
        return

    def record(_kind=None, **kw):
        if "name" not in kw:
            return
        kind = getattr(_kind, "name", str(_kind))
        targets[f"{pkg}:{kw['name']}"] = dict(kw, kind=kind, pkg=pkg)

    def glob_(patterns, exclude=()):
        out = []
        for p in patterns:
            out += [os.path.relpath(f, os.path.join(ROOT, pkg)) for f in globmod.glob(os.path.join(ROOT, pkg, p), recursive=True)]
        ex = set()
        for p in exclude:
            ex |= {os.path.relpath(f, os.path.join(ROOT, pkg)) for f in globmod.glob(os.path.join(ROOT, pkg, p), recursive=True)}
        return sorted(set(out) - ex)

    class Env(dict):
        def __missing__(self, key):
            return Rule(key)

    env = Env(load=lambda *a, **k: None, oncall=lambda *a, **k: None, fbcode_target=record,
              non_fbcode_target=lambda **k: None, glob=glob_)
    exec(compile(open(path).read(), path, "exec"), {"__builtins__": __builtins__}, env)


loaded = set()


def resolve(dep, pkg):
    if dep.startswith(":"):
        return f"{pkg}:{dep[1:]}"
    if dep.startswith("//executorch/"):
        return dep[len("//executorch/"):]
    return None  # external (torch, pypi, torchtune, ...)


closure, todo, external = [], [START], set()
while todo:
    t = todo.pop()
    if t in closure:
        continue
    pkg = t.split(":")[0]
    load_buck(pkg)
    if t not in targets:
        sys.exit(f"target not found in BUCK: {t}")
    closure.append(t)
    for d in targets[t].get("deps", []) + targets[t].get("preload_deps", []):
        r = resolve(d, pkg)
        if r is None:
            external.add(d)
        else:
            todo.append(r)

shutil.rmtree(OUT, ignore_errors=True)
for t in closure:
    spec = targets[t]
    pkg = spec["pkg"]
    base = spec.get("base_module") or "executorch." + pkg.replace("/", ".")
    for src in spec.get("srcs", []):
        if not src.endswith(".py"):
            continue
        dst = os.path.join(OUT, *base.split("."), src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(ROOT, pkg, src), dst)
print("closure:", ", ".join(sorted(closure)))
print("external deps:", ", ".join(sorted(external)))
