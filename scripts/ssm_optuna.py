#!/usr/bin/env python3
"""
scripts/ssm_optuna.py

Optuna hyperparameter search for PhasorSSM on FashionMNIST.

Calls the Julia trial script (scripts/ssm_hyperparam.jl) via subprocess,
parsing the ACCURACY= line from stdout.

Usage:
    python scripts/ssm_optuna.py                      # 50 trials, default DB
    python scripts/ssm_optuna.py --n_trials 100
    python scripts/ssm_optuna.py --storage sqlite:///ssm_study.db  # persistent DB
    python scripts/ssm_optuna.py --no-cuda             # CPU only

Search space:
    lr:           log-uniform [1e-4, 1e-2]
    epochs:       int [10, 30]
    activation:   categorical {hard, soft}
    r_lo:         uniform [0.01, 0.4]       (only if soft)
    r_hi:         r_lo + uniform [0.01, 0.5] (only if soft)
    readout_frac: uniform [0.10, 0.90]
"""

import argparse
import subprocess
import sys
import re
import os

import optuna


JULIA_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssm_hyperparam.jl")
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def run_julia_trial(params: dict, use_cuda: bool = True, verbose: bool = False) -> float:
    """Launch a single Julia trial and return the test accuracy."""
    cmd = [
        "julia", f"--project={PROJECT_ROOT}", JULIA_SCRIPT,
        "--lr",           str(params["lr"]),
        "--epochs",       str(params["epochs"]),
        "--activation",   params["activation"],
        "--r_lo",         str(params["r_lo"]),
        "--r_hi",         str(params["r_hi"]),
        "--readout_frac", str(params["readout_frac"]),
        "--hidden",       str(params.get("hidden", 128)),
        "--batchsize",    str(params.get("batchsize", 128)),
        "--seed",         str(params.get("seed", 42)),
    ]
    if not use_cuda:
        cmd.append("--no-cuda")
    if verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    # Print Julia output for debugging
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            print(f"  [julia] {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n"):
            print(f"  [julia:err] {line}")

    if result.returncode != 0:
        raise optuna.TrialPruned(f"Julia process exited with code {result.returncode}")

    # Parse ACCURACY=<value> from output
    match = re.search(r"ACCURACY=([\d.]+(?:e[+-]?\d+)?)", result.stdout)
    if match is None:
        raise optuna.TrialPruned("Could not parse ACCURACY from Julia output")

    return float(match.group(1))


def objective(trial: optuna.Trial, use_cuda: bool = True) -> float:
    # --- Sample hyperparameters ---
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 10, 30)
    activation = trial.suggest_categorical("activation", ["hard", "soft"])
    readout_frac = trial.suggest_float("readout_frac", 0.10, 0.90)

    if activation == "soft":
        r_lo = trial.suggest_float("r_lo", 0.01, 0.40)
        r_hi_offset = trial.suggest_float("r_hi_offset", 0.01, 0.50)
        r_hi = r_lo + r_hi_offset
    else:
        # Unused for hard activation, but Julia script needs values
        r_lo = 0.1
        r_hi = 0.6

    params = {
        "lr": lr,
        "epochs": epochs,
        "activation": activation,
        "r_lo": r_lo,
        "r_hi": r_hi,
        "readout_frac": readout_frac,
    }

    print(f"\n--- Trial {trial.number} ---")
    print(f"  params: {params}")

    accuracy = run_julia_trial(params, use_cuda=use_cuda)
    print(f"  accuracy: {accuracy:.4f}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Optuna search for PhasorSSM hyperparameters")
    parser.add_argument("--n_trials", type=int, default=50, help="number of Optuna trials")
    parser.add_argument("--study_name", type=str, default="phasor_ssm", help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///ssm_study.db). "
                             "Default: in-memory.")
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA for Julia trials")
    parser.add_argument("--verbose", action="store_true", help="pass --verbose to Julia trials")
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
    )

    use_cuda = not args.no_cuda

    study.optimize(
        lambda trial: objective(trial, use_cuda=use_cuda),
        n_trials=args.n_trials,
    )

    # --- Report results ---
    print("\n" + "=" * 60)
    print("  OPTUNA SEARCH COMPLETE")
    print("=" * 60)
    print(f"  Best accuracy: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Print top 5 trials
    print(f"\n  Top 5 trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1,
                           reverse=True)
    for i, t in enumerate(sorted_trials[:5]):
        print(f"    #{i+1}  acc={t.value:.4f}  params={t.params}")

    return study


if __name__ == "__main__":
    main()
