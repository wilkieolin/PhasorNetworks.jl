# PhasorSSM Hyperparameter Search

Two scripts for exploring the PhasorSSM parameter space on FashionMNIST:

- `ssm_hyperparam.jl` — Julia script that runs a single training trial and returns accuracy
- `ssm_optuna.py` — Python script that drives Optuna over the Julia trial script

Fixed choices: uniform SSM initialization, PSK (constant phase) encoding.

## Single Trial (Julia)

```bash
julia --project=. scripts/ssm_hyperparam.jl \
    --lr 0.001 --epochs 10 --activation soft \
    --r_lo 0.1 --r_hi 0.6 --readout_frac 0.25 --verbose
```

### Arguments

| Argument         | Type    | Default | Description                                      |
|------------------|---------|---------|--------------------------------------------------|
| `--lr`           | Float64 | 3e-4    | Learning rate                                    |
| `--epochs`       | Int     | 10      | Number of training epochs                        |
| `--activation`   | String  | hard    | `hard` (normalize_to_unit_circle) or `soft` (soft_normalize_to_unit_circle) |
| `--r_lo`         | Float64 | 0.1     | Soft activation lower threshold (ignored if hard)|
| `--r_hi`         | Float64 | 0.6     | Soft activation upper threshold (ignored if hard)|
| `--readout_frac` | Float64 | 0.25    | Fraction of final time steps averaged by SSMReadout |
| `--hidden`       | Int     | 128     | Hidden dimension                                 |
| `--batchsize`    | Int     | 128     | Training batch size                              |
| `--seed`         | Int     | 42      | Random seed                                      |
| `--no-cuda`      | Flag    | false   | Disable CUDA                                     |
| `--verbose`      | Flag    | false   | Print per-epoch loss and accuracy                |

Output: prints `ACCURACY=<value>` on the final line.

### Programmatic API (e.g. from Julia REPL or PyJulia)

```julia
include("scripts/ssm_hyperparam.jl")
accuracy = run_trial(; lr=1e-3, epochs=10, activation=:soft,
                       r_lo=0.1f0, r_hi=0.6f0, readout_frac=0.25f0)
```

Data is cached across calls so repeated trials don't reload FashionMNIST.

## Optuna Search (Python)

Requires: `pip install optuna`

```bash
# 50 trials, in-memory
python scripts/ssm_optuna.py

# More trials with persistent SQLite DB (resumable)
python scripts/ssm_optuna.py --n_trials 100 --storage sqlite:///ssm_study.db

# CPU only
python scripts/ssm_optuna.py --no-cuda --n_trials 20
```

### Arguments

| Argument       | Type   | Default      | Description                          |
|----------------|--------|--------------|--------------------------------------|
| `--n_trials`   | Int    | 50           | Number of Optuna trials              |
| `--study_name` | String | phasor_ssm   | Optuna study name                    |
| `--storage`    | String | None         | Storage URL (e.g. `sqlite:///study.db`). Default: in-memory |
| `--no-cuda`    | Flag   | false        | Disable CUDA for Julia trials        |
| `--verbose`    | Flag   | false        | Pass `--verbose` to Julia trials     |

### Search Space

| Parameter      | Range              | Scale        | Notes                    |
|----------------|--------------------|--------------|--------------------------|
| `lr`           | [1e-4, 1e-2]       | log-uniform  |                          |
| `epochs`       | [10, 30]           | int          |                          |
| `activation`   | {hard, soft}       | categorical  |                          |
| `r_lo`         | [0.01, 0.40]       | uniform      | Only sampled if soft     |
| `r_hi`         | r_lo + [0.01, 0.50]| uniform      | Only sampled if soft     |
| `readout_frac` | [0.10, 0.90]       | uniform      |                          |

Each trial spawns a Julia subprocess. With `--storage sqlite:///...` the study persists across runs for resumption or parallel workers.
