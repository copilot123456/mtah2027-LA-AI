"""Static inspection of a PyTorch / ultralytics ``.pt`` checkpoint — nothing is unpickled.

A ``.pt`` file is a zip archive whose ``data.pkl`` member is a pickle. Loading it with
``torch.load(..., weights_only=False)`` (what ultralytics does for its own checkpoints)
executes whatever the pickle asks for, so a checkpoint from anywhere other than your own
training run should be looked at before it is loaded. This module walks the pickle
opcode stream with :mod:`pickletools` and never calls ``pickle.load``:

* :func:`audit_imports` lists every class/function the pickle would import and flags
  everything that is not on an explicit allowlist of the names a torch / ultralytics
  detection checkpoint actually uses (tensor rebuilders, storages, ``torch.nn`` and
  ``ultralytics.nn`` module classes, ``collections.OrderedDict``, a few builtins). It
  FAILS CLOSED: an import it cannot resolve statically (a ``STACK_GLOBAL`` whose
  operands are not two plain strings, an extension-registry opcode) makes the file
  unsafe. ``safe=True`` is the precondition the SOP asks for before a foreign
  checkpoint is loaded.
* :func:`read_train_args` recovers the ``train_args`` dict ultralytics stores in every
  checkpoint (model, data, epochs, batch, imgsz, optimizer, lr0, augmentation, ...),
  which is how the baseline recipe in :mod:`training.train_rtdetr` was reconstructed
  from ``models/rtdetr_fold0_best.pt``.
* :func:`read_header` returns the ultralytics ``version`` / ``date`` stamps and
  :func:`read_train_results` the per-epoch curves saved next to them.

Only the standard library is used, so this runs on any interpreter:

    python -m training.inspect_checkpoint models/rtdetr_fold0_best.pt
    python -m training.inspect_checkpoint models/fot_v2_fold0.pt --json
"""

from __future__ import annotations

import json
import pickletools
import sys
import zipfile
from pathlib import Path

# The names a torch / ultralytics detection checkpoint legitimately imports. A whole
# root such as "torch" is NOT allowed: torch itself exposes callables (torch.load,
# torch.hub.load, ...) that a crafted pickle could invoke.
import re

ALLOWED_EXACT = frozenset({
    "collections.OrderedDict",
    "torch._utils._rebuild_tensor", "torch._utils._rebuild_tensor_v2",
    "torch._utils._rebuild_parameter", "torch._utils._rebuild_parameter_with_state",
    "torch._tensor._rebuild_from_type_v2", "torch.nn.parameter.Parameter",
    "torch.Size", "torch.device",
    "ultralytics.utils.IterableSimpleNamespace",
    "numpy.core.multiarray._reconstruct", "numpy._core.multiarray._reconstruct",
    "numpy.core.multiarray.scalar", "numpy._core.multiarray.scalar",
    "numpy.ndarray", "numpy.dtype",
    "_codecs.encode",
})
ALLOWED_PATTERNS = tuple(re.compile(p) for p in (
    r"torch\.[A-Z][A-Za-z0-9]*Storage",                       # torch.FloatStorage, ...
    r"torch\.(float|bfloat|half|int|uint|bool|complex|long|short|double)[0-9]*",
    r"torch\.nn\.modules\.[a-z_]+\.[A-Z][A-Za-z0-9_]*",       # module classes
    r"ultralytics\.nn\.modules\.[a-z_]+\.[A-Z][A-Za-z0-9_]*",
    r"ultralytics\.nn\.tasks\.[A-Z][A-Za-z0-9_]*",
    r"numpy\.dtypes\.[A-Za-z0-9]+DType",
))
# The only builtins a clean checkpoint needs. Anything else (eval, exec, getattr,
# __import__, open, object, ...) is exactly what a malicious pickle would reach for.
ALLOWED_BUILTINS = ("set", "frozenset", "dict", "list", "tuple", "slice", "range",
                    "int", "float", "bool", "str", "bytes", "bytearray", "complex")

_STRING_OPS = ("BINUNICODE", "SHORT_BINUNICODE", "UNICODE", "BINUNICODE8")
_PUT_OPS = ("BINPUT", "LONG_BINPUT")
_GET_OPS = ("BINGET", "LONG_BINGET")
_SKIP_OPS = _PUT_OPS + ("MEMOIZE",)


def _pickle_bytes(path) -> bytes:
    """The ``data.pkl`` member of a torch zip checkpoint."""
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if not members:
            raise ValueError(f"{path}: no data.pkl member — not a torch zip checkpoint")
        return archive.read(members[0])


def _resolved_ops(data: bytes) -> list[tuple[str, object]]:
    """Opcode stream as ``(kind, value)`` with memo reads resolved to their scalar.

    Pickle stores a repeated string once and refers to it by memo index afterwards
    (``BINPUT`` / ``MEMOIZE`` then ``BINGET``); without resolving those, a value such
    as the dataset path shows up as an opaque integer. Only scalars are tracked — a
    memo slot holding a container resolves to ``None`` and is reported as such.
    """
    scalar_ops = {"BININT": "int", "BININT1": "int", "BININT2": "int", "LONG1": "int",
                  "BINFLOAT": "float", "NEWTRUE": "bool", "NEWFALSE": "bool",
                  "NONE": "none"}
    memo: dict[int, tuple[str, object]] = {}
    out: list[tuple[str, object]] = []
    last: tuple[str, object] | None = None
    next_memo = 0
    for op, arg, _pos in pickletools.genops(data):
        name = op.name
        if name in _STRING_OPS:
            last = ("str", arg)
            out.append(last)
        elif name in scalar_ops:
            value = {"NEWTRUE": True, "NEWFALSE": False, "NONE": None}.get(name, arg)
            last = (scalar_ops[name], value)
            out.append(last)
        elif name in _PUT_OPS:
            if last is not None:
                memo[int(arg)] = last
        elif name == "MEMOIZE":
            if last is not None:
                memo[next_memo] = last
            next_memo += 1
        elif name in _GET_OPS:
            out.append(memo.get(int(arg), ("ref", None)))
            last = None
        elif name in ("GLOBAL", "STACK_GLOBAL"):
            out.append(("global", arg))
            last = None
        else:
            out.append(("op", name))
            last = None
    return out


def is_allowed(name: str) -> bool:
    """True when ``module.qualname`` is on the checkpoint allowlist."""
    root, _, rest = name.partition(".")
    if root in ("builtins", "__builtin__"):
        return rest in ALLOWED_BUILTINS
    if name in ALLOWED_EXACT:
        return True
    return any(pattern.fullmatch(name) for pattern in ALLOWED_PATTERNS)


# Opcodes that store to the memo without changing the stack.
_MEMO_STORE_OPS = ("MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT")
# Opcodes that import through the copyreg extension registry: not resolvable here.
_EXTENSION_OPS = ("EXT1", "EXT2", "EXT4")


def audit_imports(path) -> dict:
    """Every import the pickle would perform, and whether the file may be loaded.

    Returns ``{"imports": [...], "outside_allowlist": [...], "unresolved": [...],
    "safe": bool}``. Imports come from ``GLOBAL`` and ``INST`` (module and name in the
    opcode argument) and from ``STACK_GLOBAL`` (module and name are the two values on
    top of the stack). A ``STACK_GLOBAL`` is resolved only when the two value-pushing
    opcodes right before it are plain strings — literal, or read back from the memo;
    anything else (an intervening ``POP``, a computed value, ...) is reported in
    ``unresolved``, as is any extension-registry opcode. ``safe`` is True only when
    every import is allowed (:func:`is_allowed`) and nothing is unresolved.
    """
    data = _pickle_bytes(path)
    imports: set[str] = set()
    unresolved: list[str] = []
    memo: dict[int, object] = {}
    pushes: list[object] = []          # value-pushing ops, newest last (str or None)
    next_memo = 0
    for op, arg, pos in pickletools.genops(data):
        name = op.name
        if name in _MEMO_STORE_OPS:
            index = next_memo if name == "MEMOIZE" else int(arg)
            if name == "MEMOIZE":
                next_memo += 1
            memo[index] = pushes[-1] if pushes else None
            continue
        if name == "GLOBAL":
            imports.add(str(arg).replace(" ", "."))
            pushes.append(None)
        elif name == "INST":
            imports.add(str(arg).replace(" ", "."))
            pushes.append(None)
        elif name == "STACK_GLOBAL":
            operands = pushes[-2:]
            if len(operands) == 2 and all(isinstance(v, str) for v in operands):
                imports.add(".".join(operands))
            else:
                unresolved.append(f"STACK_GLOBAL at byte {pos} with non-literal operands")
            pushes.append(None)
        elif name in _EXTENSION_OPS:
            unresolved.append(f"{name} at byte {pos} (copyreg extension registry)")
            pushes.append(None)
        elif name in _STRING_OPS:
            pushes.append(str(arg))
        elif name in ("BINGET", "LONG_BINGET", "GET"):
            pushes.append(memo.get(int(arg)))
        elif name in ("PROTO", "FRAME", "STOP"):
            continue
        else:
            pushes.append(None)      # any other op: the top of the stack is not a string

    outside = sorted(n for n in imports if not is_allowed(n))
    return {"imports": sorted(imports), "outside_allowlist": outside,
            "unresolved": unresolved, "safe": not outside and not unresolved}


def _dict_after(ops: list[tuple[str, object]], key: str, stop_keys: tuple[str, ...]):
    """Flat ``key -> scalar`` pairs following the string ``key`` until a stop key."""
    start = next((i for i, (kind, value) in enumerate(ops)
                  if kind == "str" and value == key), None)
    if start is None:
        return None
    pairs: dict[str, object] = {}
    pending: str | None = None
    for kind, value in ops[start + 1:]:
        if kind in ("op", "global"):
            continue
        if kind == "str" and pending is None:
            if value in stop_keys:
                break
            pending = str(value)
            continue
        if pending is not None:
            pairs[pending] = value
            pending = None
    return pairs


def read_train_args(path) -> dict | None:
    """The ultralytics ``train_args`` dict, or None when the file has none.

    A bare ``state_dict`` (such as ``fot_v2_fold0.pt``) has no ``train_args``.
    """
    ops = _resolved_ops(_pickle_bytes(path))
    return _dict_after(ops, "train_args", stop_keys=("train_metrics", "train_results"))


def read_header(path) -> dict:
    """Top-level stamps ultralytics writes: ``date``, ``version``, ``license``."""
    ops = _resolved_ops(_pickle_bytes(path))
    header = {}
    for key in ("date", "version", "license"):
        index = next((i for i, (kind, value) in enumerate(ops[:200])
                      if kind == "str" and value == key), None)
        if index is not None and index + 1 < len(ops) and ops[index + 1][0] == "str":
            header[key] = ops[index + 1][1]
    return header


def read_train_results(path) -> dict | None:
    """Per-epoch curves (``metrics/mAP50(B)``, losses, lr) saved in the checkpoint."""
    ops = _resolved_ops(_pickle_bytes(path))
    start = next((i for i, (kind, value) in enumerate(ops)
                  if kind == "str" and value == "train_results"), None)
    if start is None:
        return None
    curves: dict[str, list] = {}
    current: str | None = None
    for kind, value in ops[start + 1:]:
        if kind == "str":
            if value == "git":
                break
            current = str(value)
            curves[current] = []
        elif kind in ("int", "float") and current is not None:
            curves[current].append(value)
    return {k: v for k, v in curves.items() if v}


def state_dict_keys(path) -> list[str]:
    """Parameter names of a bare ``state_dict`` checkpoint (dotted strings only)."""
    data = _pickle_bytes(path)
    keys = []
    for op, arg, _pos in pickletools.genops(data):
        if op.name in _STRING_OPS and "." in str(arg) and " " not in str(arg):
            if str(arg).rsplit(".", 1)[-1] in ("weight", "bias", "in_proj_weight",
                                              "in_proj_bias"):
                keys.append(str(arg))
    return keys


def describe(path) -> dict:
    """Everything this module can say about one checkpoint, as a JSON-ready dict."""
    path = Path(path)
    report = {"file": str(path), "bytes": path.stat().st_size,
              "audit": audit_imports(path), "header": read_header(path)}
    train_args = read_train_args(path)
    if train_args is not None:
        report["train_args"] = train_args
        report["train_results"] = read_train_results(path)
    else:
        report["state_dict_keys"] = state_dict_keys(path)
    return report


def main(argv: list[str] | None = None) -> int:
    """Print the static report for each checkpoint; exit 1 if any audit is not safe."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m training.inspect_checkpoint",
        description="Static (no unpickling) audit and metadata dump of .pt checkpoints.")
    parser.add_argument("checkpoints", nargs="+", help=".pt files to inspect")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    reports = [describe(p) for p in args.checkpoints]
    if args.json:
        print(json.dumps(reports, indent=2, default=str))
    else:
        for report in reports:
            audit = report["audit"]
            print(f"== {report['file']} ({report['bytes'] / 1e6:.1f} MB)")
            print(f"   import audit : {'SAFE' if audit['safe'] else 'NOT SAFE'}"
                  f" ({len(audit['imports'])} imports,"
                  f" {len(audit['outside_allowlist'])} outside the allowlist,"
                  f" {len(audit['unresolved'])} unresolved)")
            for name in audit["outside_allowlist"]:
                print(f"     !! {name}")
            for item in audit["unresolved"]:
                print(f"     !! {item}")
            for key, value in report["header"].items():
                print(f"   {key:13s}: {value}")
            if "train_args" in report:
                print("   train_args   :")
                for key, value in report["train_args"].items():
                    print(f"     {key:16s} = {value!r}")
                results = report.get("train_results") or {}
                for key in ("metrics/mAP50(B)", "metrics/mAP50-95(B)",
                            "metrics/precision(B)", "metrics/recall(B)"):
                    if key in results:
                        print(f"   {key:22s}: {results[key]}")
            else:
                keys = report["state_dict_keys"]
                print(f"   bare state_dict with {len(keys)} weight/bias tensors"
                      f" (first: {keys[:3]})")
    return 0 if all(r["audit"]["safe"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
