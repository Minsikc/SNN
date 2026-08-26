"""End-to-end: the yaml -> main_unified -> BasicExperiment pipeline builds the new
EpropRSNN model (LIF / ALIF, software / mock hardware) and trains a couple of
epochs. Runs in a temporary results dir; no serial port."""
import os
import subprocess
import sys

import yaml

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cfg_name, tmp_path, epochs=2, overrides=None):
    src = os.path.join(PKG_ROOT, "configs", cfg_name)
    cfg = yaml.safe_load(open(src, encoding="utf-8"))
    cfg["experiment"]["results_dir"] = str(tmp_path)
    if "grad_log_path" in cfg["model"].get("hardware", {}):
        cfg["model"]["hardware"]["grad_log_path"] = str(tmp_path / "grad.csv")
    for k, v in (overrides or {}).items():
        d = cfg
        *ks, last = k.split(".")
        for kk in ks:
            d = d.setdefault(kk, {})
        d[last] = v
    out = os.path.join(PKG_ROOT, "configs", f"_tmp_{cfg_name}")
    yaml.safe_dump(cfg, open(out, "w", encoding="utf-8"))
    try:
        p = subprocess.run([sys.executable, "main_unified.py", "--config", os.path.basename(out),
                            "--epochs", str(epochs)], cwd=PKG_ROOT, capture_output=True, text=True,
                           timeout=600)
    finally:
        os.remove(out)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout


def test_alif_digital_config_runs(tmp_path):
    out = _run("eprop_alif_digital.yaml", tmp_path)
    assert "Experiment completed successfully" in out
    assert out.count("Epoch ") == 2


def test_alif_mock_hardware_config_runs_and_logs_gradient(tmp_path):
    out = _run("eprop_alif_mock.yaml", tmp_path)
    assert "Experiment completed successfully" in out
    csv_path = tmp_path / "grad.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) == 1 + 2 * 16          # header + 2 epochs x 4x4 cells


def test_lif_config_via_eprop_core_matches_legacy_first_epoch(tmp_path):
    """Same yaml through the legacy RSNN_eprop_HW_forward (mock) and through EpropRSNN
    (mock) gives the same epoch-1 loss (identical init via training.init_seed)."""
    import re
    legacy = _run("eprop_4x4_mock.yaml", tmp_path, epochs=1)
    new = _run("eprop_4x4_mock.yaml", tmp_path, epochs=1,
               overrides={"model.type": "EpropRSNN", "model.neuron": {"kind": "lif"},
                          "model.grad_chain": {}})
    pat = re.compile(r"Epoch 1/1, Loss: ([0-9.]+)")
    assert pat.search(legacy).group(1) == pat.search(new).group(1)
