"""Reference click times from the transponder_model toy network.

Runs transponder_model/src/tests/toy_example.py (gitlab.com/spcnc/transponder_model)
for each configuration its CI job `toy-tests` runs, and writes every node's
input- and output-nTron click times to <out>.json and <out>.csv. demos/device_studies/ntron_toy_validation.jl
replays the same networks on NTronTransponder cells in a DeviceReservoir and
compares.

    python3 demos/device_studies/ntron_toy_reference.py <path/to/toy_example.py> <out.json>

matplotlib is stubbed out (the toy only uses it to save plots), and the toy's
per-RHS-call prints are discarded.
"""
import contextlib
import io
import json
import os
import runpy
import sys
import types
from unittest import mock

import numpy  # noqa: F401  -- imported before sys.modules is patched, so it survives between runs
import scipy.integrate  # noqa: F401

# The seven CI invocations (.gitlab-ci.yml, job toy-tests), minus --output.
BASE = ["--Ib1init", "12", "--I1sw", "13.5", "--Ib2init", "23", "--I2sw", "24", "--split", "0.3"]
CONFIGS = {
    "toy1_n4ob23":               ["--dt", "35", "--ib1", "2", "13", "--ib1", "4", "11", "--ib2", "4", "23"],
    "toy1_dt35":                 ["--dt", "35", "--ib1", "2", "13", "--ib1", "4", "11", "--ib2", "4", "22.6"],
    "toy1_dt30":                 ["--dt", "30", "--ib1", "2", "13", "--ib1", "4", "11", "--ib2", "4", "22.6"],
    "toy1_dt25":                 ["--dt", "25", "--ib1", "2", "13", "--ib1", "4", "11", "--ib2", "4", "22.6"],
    "toy1_15":                   ["--dt", "15", "--ib1", "2", "13", "--ib1", "4", "11", "--ib2", "4", "22.6"],
    "toy1_15_n2ob23.4":          ["--dt", "15", "--ib1", "2", "13", "--ib1", "4", "11", "--ib2", "4", "22.6",
                                  "--ib2", "2", "23.4"],
    "toy1_15_n2ob23.4_n3ob22.95": ["--dt", "15", "--ib1", "2", "13", "--ib1", "4", "11", "--ib2", "4", "22.6",
                                   "--ib2", "2", "23.4", "--ib2", "3", "22.95"],
}


def run_one(toy_path, args):
    stub = types.ModuleType("matplotlib")
    stub.pyplot = mock.MagicMock()
    stub.pyplot.subplots.return_value = (mock.MagicMock(), mock.MagicMock())
    with mock.patch.dict(sys.modules, {"matplotlib": stub, "matplotlib.pyplot": stub.pyplot}), \
         mock.patch.object(sys, "argv", ["toy_example.py"] + args), \
         contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path(toy_path, run_name="toy")   # not __main__: defines main() without running it
        g["main"]()                                     # mutates g's click lists in place
    return {
        "args": args,
        "delay_ns": (g["delay"] * 1e9).tolist(),
        "bias1_uA": (g["transponder_bias1"] * 1e6).tolist(),
        "bias2_uA": (g["transponder_bias2"] * 1e6).tolist(),
        "sensor_t_ns": [1.0, 1.0 + g["dt"] * 1e9],
        "input_clicks_ns": [[t * 1e9 for t in c] for c in g["input_click_times"]],
        "output_clicks_ns": [[t * 1e9 for t in c] for c in g["output_click_times"]],
    }


def main():
    toy_path, out = os.path.abspath(sys.argv[1]), sys.argv[2]
    sys.path.insert(0, os.path.dirname(os.path.dirname(toy_path)))   # src/, for `import spcnc...`
    only = sys.argv[3:]                                  # optional subset of config names
    res = {}
    for name, extra in CONFIGS.items():
        if only and name not in only:
            continue
        res[name] = run_one(toy_path, BASE + extra)
        print(name, "outputs:", [[round(t, 3) for t in c] for c in res[name]["output_clicks_ns"]], flush=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    # The same, long-format, for Julia (CSV.jl): config,kind,a,b,value
    #   delay a→b ns · bias1/bias2 of node a µA · sensor a fires at value ns ·
    #   in/out click of node a at value ns. Nodes are 1-based here.
    rows = ["config,kind,a,b,value"]
    for name, r in res.items():
        for i, row in enumerate(r["delay_ns"]):
            rows += [f"{name},delay,{i+1},{j+1},{d}" for j, d in enumerate(row) if d > 0]
        for i, (b1, b2) in enumerate(zip(r["bias1_uA"], r["bias2_uA"])):
            rows += [f"{name},bias1,{i+1},0,{b1}", f"{name},bias2,{i+1},0,{b2}"]
        rows += [f"{name},sensor,{i+1},0,{t}" for i, t in enumerate(r["sensor_t_ns"])]
        for kind in ("input", "output"):
            for i, c in enumerate(r[kind + "_clicks_ns"]):
                rows += [f"{name},{kind},{i+1},0,{t}" for t in c]
    with open(os.path.splitext(out)[0] + ".csv", "w") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
