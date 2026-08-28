#!/usr/bin/env python3
"""Query the run registry (results/registry.jsonl or $SNN_REGISTRY).

    python scripts/analysis/registry_table.py                              # all runs, grouped by task/condition
    python scripts/analysis/registry_table.py --task xor --group condition neuron.kind train_hidden \
        --metrics metrics.best_acc metrics.first_perfect_epoch --agg mean
    python scripts/analysis/registry_table.py --filter neuron.kind=alif --filter chain.eligibility=full
    python scripts/analysis/registry_table.py --list --task teacher_student   # one line per run
    python scripts/analysis/registry_table.py --latest-only                    # drop superseded re-runs
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # package root

from eprop import registry  # noqa: E402


def _parse(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(v.lower(), v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=None, help="registry file (default results/registry.jsonl or $SNN_REGISTRY)")
    ap.add_argument("--task", default=None)
    ap.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted key filter, repeatable (e.g. neuron.kind=alif, metrics.best_acc=1.0)")
    ap.add_argument("--group", nargs="+", default=["task", "condition", "neuron.kind", "train_hidden"])
    ap.add_argument("--metrics", nargs="+", default=["metrics.best_loss"])
    ap.add_argument("--agg", default="mean", choices=["mean", "median", "min", "max"])
    ap.add_argument("--latest-only", action="store_true", help="keep only the newest entry per run_id")
    ap.add_argument("--list", action="store_true", help="print one line per run instead of a grouped table")
    args = ap.parse_args()

    filters = {}
    if args.task:
        filters["task"] = args.task
    for f in args.filter:
        k, _, v = f.partition("=")
        filters[k] = _parse(v)
    rows = registry.load(args.path, **filters)
    if not rows:
        print(f"no entries in {registry.registry_path(args.path)} matching {filters}")
        return 1

    if args.list:
        cols = ["run_id", "timestamp", "git_commit", "entry_point", "task", "condition", "neuron.kind",
                "neuron.beta", "neuron.n_adaptive", "chain.eligibility", "train_hidden", "seed", "epochs", "lr",
                *args.metrics]
        print("| " + " | ".join(c.replace("metrics.", "") for c in cols) + " |")
        print("|" + "---|" * len(cols))
        for r in rows:
            print("| " + " | ".join(registry._fmt(registry.get(r, c)) for c in cols) + " |")
    else:
        print(registry.table(rows, group=args.group, metrics=args.metrics, agg=args.agg,
                             latest_only=args.latest_only))
    print(f"\n{len(rows)} runs from {registry.registry_path(args.path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
