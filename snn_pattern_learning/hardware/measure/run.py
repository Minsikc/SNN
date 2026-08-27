"""Run a legacy measurement script from inside the data directory.

The scripts under ``hardware/measure/0*_*/`` read and write their CSV/NPZ
files relative to the current working directory (``glob("*_seq_attractor.csv")``,
hard-coded ``"2026-08-10_14-29_CycleTest_Data.csv"`` ...). This runner
changes into :func:`common.data_dir` (``hardware/measure/data`` or
``$SNN_MEASURE_DATA``), puts the script's own folder on ``sys.path`` (so
same-folder imports such as ``import fit_percell_cycletest`` work) and the
package root as well (for ``from hardware import MemristorInterface``), then
executes the script with the remaining arguments.

Usage (from ``snn_pattern_learning/``)::

    python -m hardware.measure.run --list
    python -m hardware.measure.run 03_read_disturbance/fit_read_disturb_modelB.py
    python -m hardware.measure.run 02_grad_fidelity_uv/uv_random_signed.py --port COM4 --trials 40
    python -m hardware.measure.run --data-dir D:/other/data 01_potdep_curve/plot_cycle_test.py x.csv

Hardware scripts talk to the Arduino Due on COM4 (default in each script);
they are NOT run by the test-suite.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(os.path.dirname(HERE))


def list_scripts():
    for grp in sorted(d for d in os.listdir(HERE) if d[:2].isdigit()):
        print(grp)
        for f in sorted(os.listdir(os.path.join(HERE, grp))):
            if f.endswith(".py"):
                print("   ", f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", nargs="?", help="<group>/<file>.py relative to hardware/measure")
    ap.add_argument("--data-dir", default=None, help="override the data directory")
    ap.add_argument("--list", action="store_true", help="list available scripts")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="arguments passed to the script")
    ns = ap.parse_args(argv)

    if ns.list or not ns.script:
        list_scripts()
        return 0

    script = os.path.join(HERE, ns.script)
    if not os.path.isfile(script):
        ap.error(f"no such script: {script}")

    from . import common
    if ns.data_dir:
        os.environ["SNN_MEASURE_DATA"] = os.path.abspath(ns.data_dir)
    data = common.data_dir()

    sys.path.insert(0, os.path.dirname(script))
    sys.path.insert(0, PKG_ROOT)
    sys.argv = [script] + ns.args
    os.chdir(data)
    print(f"[measure.run] cwd = {data}")
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
