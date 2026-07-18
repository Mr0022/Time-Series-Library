# Bayesian hyper-parameter tuning (Optuna / TPE)

`tune_optuna.py` runs Bayesian optimization (Optuna's TPE sampler) over the
11 realized-volatility forecasting models, using the **validation aggregated
MSE** as the objective on the `--aggregate_mean` (Option 1) target
`log(mean(RV))`.

Models: `DLinear, PatchTST, iTransformer, TimesNet, MSGNet, TimeMixer, FITS,
WFTNet, TSLANet, ModernTCN, AdaWaveNet`.

## Install

```bash
pip install -r requirements.txt
pip install -r tuning/requirements-tuning.txt   # adds optuna>=3.4
```

## Run

```bash
# all 11 models, 40 trials each, weekly horizon (H=5), auto GPU/CPU
python tuning/tune_optuna.py --models all --n_trials 40 --pred_len 5

# just the cheap ones, more trials, on GPU
python tuning/tune_optuna.py --models FITS,DLinear --n_trials 80 --device cuda

# monthly horizon, persist/resume studies in a SQLite DB
python tuning/tune_optuna.py --pred_len 22 --storage tuning/results/rv_tuning.db
```

Key flags: `--n_trials`, `--pred_len` (aggregation horizon H), `--seq_len`
(66 by default), `--train_epochs`/`--patience` (fixed, not searched),
`--device {auto,cpu,cuda}`, `--timeout` (per-model seconds), `--storage`
(SQLite to resume).

## What it searches

Per-model search spaces and priors are defined in the `suggest_*` functions
(learning rate log-uniform, batch size categorical, dropout, plus each
model's structural knobs). Objective = minimize validation aggregated MSE of
the early-stopped checkpoint.

### `seq_len = 66` note

`66 = 2·3·11` halves only once, so two models get one dimension held fixed at
this length (encoded in `MODEL_FIXED`):

- **TimeMixer** → `down_sampling_layers = 1`
- **AdaWaveNet** → `lifting_levels = 1`

Use `--seq_len 64` (=2⁶) to keep those flexible; the fixes are harmless there.

### Constraints (encoded so no trial crashes)

`n_heads | d_model`, `d_ff ≥ d_model`, `moving_avg` odd, `patch_size ≤ seq_len`,
`patch_stride ≤ patch_size`. Univariate-degenerate knobs are held fixed to keep
each search to ~4–7 dimensions (iTransformer `n_heads=4`, MSGNet graph defaults,
PatchTST head set, AdaWaveNet `n_clusters=1`).

Two knobs are **not CLI-wired** in this repo and are therefore left at their
in-model defaults (would need a `run.py` arg to search): ModernTCN
`ffn_ratio/large_size/small_size/num_blocks`, TSLANet `ICB/ASB/adaptive_filter`,
and PatchTST `patch_len/stride`.

## Output

Written to `tuning/results/`:

- `<Model>_best.json` — best value, searched params, full effective config,
  and a ready-to-run `python run.py …` command reproducing the best config.
- `<Model>_trials.csv` — every trial (params + value) for inspection.
- `summary.json` — best validation MSE per model, ascending.

Reuses `Exp_Long_Term_Forecast.train()` unchanged, so training/early-stopping
semantics are identical to `run.py`; checkpoints are written under
`--checkpoints` and deleted per trial.
