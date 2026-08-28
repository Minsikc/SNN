"""Run registry: one JSON line per run with a stable schema, shared by every
entry point (``eprop.run_condition``, ``eprop.xor.run_xor``, ``main_unified``).

    results/registry.jsonl          (override with $SNN_REGISTRY)

Schema (one dict per line)::

    run_id        sha1 of the resolved configuration (task, condition, neuron,
                  chain, hw, train_hidden, seed, epochs, lr) -> identical
                  settings get identical ids, so re-runs are easy to spot
    timestamp     ISO-8601 local time
    git_commit    short hash of the SNN repo at run time (or None)
    entry_point   "run_condition" | "run_xor" | "main_unified"
    task          "teacher_student" | "xor" | dataset type
    condition     bptt | digital | digital_mock | frozen(_wout) | analog
    neuron, chain, hw, task_cfg   resolved config dicts (hw may be None)
    train_hidden  bool
    seed, epochs, lr
    metrics       flat dict of summary numbers (best_loss, best_epoch, best_acc,
                  first_perfect_epoch, fidelity_pooled, ... whatever the run has)
    curves_path   file holding the per-epoch curves (sweep JSON / results dir)
    seconds       wall time
    note          free text

Reading back::

    from eprop.registry import load, table
    rows = load()
    print(table(rows, group=["task", "condition", "neuron.kind"], metrics=["best_loss"]))
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_PKG_ROOT, "results", "registry.jsonl")


def registry_path(path: Optional[str] = None) -> str:
    return path or os.environ.get("SNN_REGISTRY") or DEFAULT_PATH


def _jsonable(x):
    if hasattr(x, "__dataclass_fields__"):
        return {k: _jsonable(getattr(x, k)) for k in x.__dataclass_fields__}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, float) and x != x:      # NaN -> None (valid JSON)
        return None
    return x


def git_commit(cwd: str = _PKG_ROOT) -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, timeout=5, check=True).stdout.strip() or None
    except Exception:
        return None


def make_run_id(task: str, condition: str, neuron, chain, hw, task_cfg, train_hidden: bool,
                seed: int, epochs: int, lr: float) -> str:
    key = _jsonable(dict(task=task, condition=condition, neuron=neuron, chain=chain, hw=hw,
                         task_cfg=task_cfg, train_hidden=train_hidden, seed=seed, epochs=epochs, lr=lr))
    return hashlib.sha1(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]


def make_entry(*, entry_point: str, task: str, condition: str, neuron, chain, hw, task_cfg,
               train_hidden: bool, seed: int, epochs: int, lr: float, metrics: Dict[str, Any],
               curves_path: Optional[str] = None, seconds: Optional[float] = None,
               note: str = "") -> Dict[str, Any]:
    neuron, chain, hw, task_cfg = map(_jsonable, (neuron, chain, hw, task_cfg))
    return OrderedDict(
        run_id=make_run_id(task, condition, neuron, chain, hw, task_cfg, train_hidden, seed, epochs, lr),
        timestamp=_dt.datetime.now().isoformat(timespec="seconds"),
        git_commit=git_commit(),
        entry_point=entry_point, task=task, condition=condition,
        neuron=neuron, chain=chain, hw=hw, task_cfg=task_cfg,
        train_hidden=bool(train_hidden), seed=int(seed), epochs=int(epochs), lr=float(lr),
        metrics=_jsonable(metrics), curves_path=curves_path,
        seconds=None if seconds is None else round(float(seconds), 2), note=note,
    )


def record(entry: Dict[str, Any], path: Optional[str] = None) -> str:
    """Append one entry; returns the registry path."""
    p = registry_path(path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=False) + "\n")
    return p


def load(path: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
    """Load all entries; ``filters`` match top-level or dotted keys
    (``load(task="xor", **{"neuron.kind": "alif"})``)."""
    p = registry_path(path)
    if not os.path.exists(p):
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if all(get(r, k) == v for k, v in filters.items())]


def get(row: Dict[str, Any], key: str, default=None):
    """Dotted lookup: ``get(row, "neuron.beta")``, ``get(row, "metrics.best_loss")``."""
    cur: Any = row
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    return str(v)


def table(rows: Iterable[Dict[str, Any]], group: Sequence[str], metrics: Sequence[str] = ("metrics.best_loss",),
          agg: str = "mean", latest_only: bool = False) -> str:
    """Markdown table: one line per distinct ``group`` key, ``metrics`` aggregated
    (mean / median / min / max, plus n). ``latest_only`` keeps the most recent
    entry per run_id (re-runs of identical settings)."""
    rows = list(rows)
    if latest_only:
        seen: Dict[str, Dict] = {}
        for r in rows:
            seen[r["run_id"]] = r          # file order == chronological
        rows = list(seen.values())
    groups: "OrderedDict[tuple, List[Dict]]" = OrderedDict()
    for r in rows:
        groups.setdefault(tuple(_fmt(get(r, g)) for g in group), []).append(r)

    def reduce(vals: List[float]):
        vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
        if not vals:
            return None
        if agg == "mean":
            return sum(vals) / len(vals)
        if agg == "median":
            s = sorted(vals); m = len(s) // 2
            return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])
        if agg == "min":
            return min(vals)
        if agg == "max":
            return max(vals)
        raise ValueError(agg)

    head = [*group, "n", *[f"{agg}({m.replace('metrics.', '')})" for m in metrics]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for key, rs in groups.items():
        cells = [*key, str(len(rs)), *[_fmt(reduce([get(r, m) for r in rs])) for m in metrics]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
