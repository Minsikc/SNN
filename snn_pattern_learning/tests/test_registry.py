"""Run registry: stable ids, append/load/filter, table, and the three entry-point hooks."""
import json
import os
import subprocess
import sys

import pytest
import yaml

from eprop import GradChainConfig, HardwareReadoutConfig, NeuronConfig, TaskConfig, registry, run_condition
from eprop.xor import run_xor

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_run_id_is_deterministic_and_config_sensitive():
    n, c, t = NeuronConfig(kind="alif", beta=0.5, n_adaptive=2), GradChainConfig(), TaskConfig()
    a = registry.make_run_id("teacher_student", "digital", n, c, None, t, True, 0, 50, 0.05)
    b = registry.make_run_id("teacher_student", "digital", n, c, None, t, True, 0, 50, 0.05)
    assert a == b and len(a) == 12
    assert a != registry.make_run_id("teacher_student", "digital", n, c, None, t, True, 1, 50, 0.05)
    assert a != registry.make_run_id("teacher_student", "digital", NeuronConfig(kind="alif", beta=0.5, n_adaptive=3),
                                     c, None, t, True, 0, 50, 0.05)
    assert a != registry.make_run_id("teacher_student", "digital", n, GradChainConfig(eligibility="truncated"),
                                     None, t, True, 0, 50, 0.05)


def test_record_load_filter_table(tmp_path):
    p = str(tmp_path / "reg.jsonl")
    for seed, loss in [(0, 0.9), (1, 0.6), (2, 1.2)]:
        e = registry.make_entry(entry_point="t", task="teacher_student", condition="digital",
                                neuron=NeuronConfig(), chain=GradChainConfig(), hw=None, task_cfg=TaskConfig(),
                                train_hidden=True, seed=seed, epochs=50, lr=0.05,
                                metrics=dict(best_loss=loss, best_acc=None), curves_path="x.json")
        registry.record(e, p)
    e2 = registry.make_entry(entry_point="t", task="xor", condition="bptt", neuron=NeuronConfig(kind="alif", beta=0.5),
                             chain=GradChainConfig(), hw=None, task_cfg={}, train_hidden=True, seed=0, epochs=60,
                             lr=0.05, metrics=dict(best_loss=1.0, best_acc=0.75, fidelity_mean=float("nan")))
    registry.record(e2, p)

    rows = registry.load(p)
    assert len(rows) == 4 and rows[0]["run_id"] != rows[1]["run_id"]
    assert registry.load(p, task="xor")[0]["neuron"]["kind"] == "alif"
    assert len(registry.load(p, **{"neuron.kind": "lif"})) == 3
    assert rows[3]["metrics"]["fidelity_mean"] is None          # NaN -> null (valid JSON)
    assert all(json.loads(l) for l in open(p))                   # every line parses

    md = registry.table(rows, group=["task", "condition"], metrics=["metrics.best_loss"], agg="mean")
    assert "| teacher_student | digital | 3 | 0.9 |" in md
    md = registry.table(rows, group=["task"], metrics=["metrics.best_loss"], agg="median")
    assert "| teacher_student | 3 | 0.9 |" in md


def test_latest_only_dedups_reruns(tmp_path):
    p = str(tmp_path / "reg.jsonl")
    for loss in (1.0, 0.5):
        registry.record(registry.make_entry(entry_point="t", task="x", condition="digital", neuron=NeuronConfig(),
                                            chain=GradChainConfig(), hw=None, task_cfg={}, train_hidden=True,
                                            seed=0, epochs=1, lr=0.1, metrics=dict(best_loss=loss)), p)
    rows = registry.load(p)
    assert "| x | 2 | 0.75 |" in registry.table(rows, ["task"])
    assert "| x | 1 | 0.5 |" in registry.table(rows, ["task"], latest_only=True)


def test_run_condition_and_run_xor_record(tmp_path, monkeypatch):
    p = str(tmp_path / "reg.jsonl")
    monkeypatch.setenv("SNN_REGISTRY", p)
    r = run_condition("digital", 2, 0.05, seed=0, record=True, curves_path="sweep.json", note="unit")
    assert r["registry_path"] == p
    r2 = run_xor("digital_mock", 2, 0.15, seed=0, record=True)
    rows = registry.load(p)
    assert [x["entry_point"] for x in rows] == ["run_condition", "run_xor"]
    assert rows[0]["metrics"]["best_loss"] == pytest.approx(r["best_loss"])
    assert rows[0]["curves_path"] == "sweep.json" and rows[0]["note"] == "unit"
    assert rows[1]["hw"]["use_mock"] is True and rows[1]["train_hidden"] is False
    assert rows[1]["metrics"]["best_acc"] == r2["best_acc"]
    # not recorded by default
    run_condition("digital", 1, 0.05, seed=0)
    assert len(registry.load(p)) == 2


def test_main_unified_records(tmp_path, monkeypatch):
    p = str(tmp_path / "reg.jsonl")
    cfg = yaml.safe_load(open(os.path.join(PKG_ROOT, "configs", "eprop_alif_mixed.yaml"), encoding="utf-8"))
    cfg["experiment"]["results_dir"] = str(tmp_path / "res")
    out = os.path.join(PKG_ROOT, "configs", "_tmp_registry_test.yaml")
    yaml.safe_dump(cfg, open(out, "w", encoding="utf-8"))
    env = dict(os.environ, SNN_REGISTRY=p)
    try:
        pr = subprocess.run([sys.executable, "main_unified.py", "--config", os.path.basename(out), "--epochs", "2"],
                            cwd=PKG_ROOT, capture_output=True, text=True, timeout=600, env=env)
    finally:
        os.remove(out)
    assert pr.returncode == 0, pr.stdout[-2000:] + pr.stderr[-2000:]
    rows = registry.load(p)
    assert len(rows) == 1
    e = rows[0]
    assert e["entry_point"] == "main_unified" and e["condition"] == "digital"
    assert e["neuron"]["kind"] == "alif" and e["neuron"]["n_adaptive"] == 2
    assert e["epochs"] == 2 and len(e["history"]) == 2
    assert e["metrics"]["best_loss"] == pytest.approx(min(h["train_loss"] for h in e["history"]))
