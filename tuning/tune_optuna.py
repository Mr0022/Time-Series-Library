#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Bayesian hyper-parameter optimization (Optuna / TPE) for the 11 realized-
# volatility forecasting models in this repo.
#
# Models: DLinear, PatchTST, iTransformer, TimesNet, MSGNet, TimeMixer,
#         FITS, WFTNet, TSLANet, ModernTCN, AdaWaveNet
#
# Task   : long_term_forecast, univariate ln_RV, --aggregate_mean (Option 1),
#          i.e. the model forecasts a single value log(mean(RV)) over the next
#          `pred_len` days. The BO objective is the *validation* aggregated MSE
#          of the best (early-stopped) checkpoint.
#
# Design notes
# ------------
# * No changes to the repo's model / run.py code. This driver rebuilds the
#   exact argparse Namespace that run.py would produce, then reuses
#   Exp_Long_Term_Forecast.train() so training semantics stay identical.
# * seq_len is fixed to 66 by default. `66 = 2*3*11` halves only once, so two
#   models get a constrained (fixed) dimension at 66:
#       - TimeMixer   -> down_sampling_layers = 1
#       - AdaWaveNet  -> lifting_levels = 1
#   (Use --seq_len 64 to keep those flexible; the fixes below are harmless there.)
# * Search-space / crash constraints are encoded so no trial dies on a bad
#   config: n_heads | d_model, d_ff >= d_model, moving_avg odd, patch sizes
#   <= seq_len, patch_stride <= patch_size, plus the univariate-degenerate
#   knobs (iTransformer n_heads, MSGNet graph, PatchTST heads) held fixed to
#   keep each search to ~4-7 dims.
#
# Usage
# -----
#   pip install optuna>=3.4          # (see tuning/requirements-tuning.txt)
#   python tuning/tune_optuna.py --models all --n_trials 40 --pred_len 5
#   python tuning/tune_optuna.py --models FITS,DLinear --n_trials 60 --device cuda
#
# Results are written to tuning/results/ (per-model best JSON + summary + an
# optuna study SQLite DB if --storage is passed).
# ---------------------------------------------------------------------------
import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

try:
    import optuna
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "optuna is required: pip install -r tuning/requirements-tuning.txt "
        "(or pip install 'optuna>=3.4')"
    ) from e

# Run everything from the repo root so relative paths ('models', './data/')
# resolve exactly like `python run.py` does.
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

BATCH_SIZES = [16, 32, 64, 128]


# ---------------------------------------------------------------------------
# run.py argparse, reconstructed verbatim (defaults must match run.py exactly).
# We build the args by parsing an argv list, so store_true flags, types and
# defaults behave identically to a real CLI invocation.
# ---------------------------------------------------------------------------
def make_run_parser():
    parser = argparse.ArgumentParser(description='RV tuning (mirrors run.py)')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1)
    parser.add_argument('--model_id', type=str, required=True, default='test')
    parser.add_argument('--model', type=str, required=True, default='Autoformer')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    parser.add_argument('--inverse', action='store_true', default=False)
    parser.add_argument('--aggregate_mean', action='store_true', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25)

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25)

    # model define
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--tv_dt', type=int, default=0)
    parser.add_argument('--tv_B', type=int, default=0)
    parser.add_argument('--tv_C', type=int, default=0)
    parser.add_argument('--use_D', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--seg_len', type=int, default=96)

    # optimization
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    # metrics (dtw)
    parser.add_argument('--use_dtw', action='store_true', default=False)

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--jitter', default=False, action="store_true")
    parser.add_argument('--scaling', default=False, action="store_true")
    parser.add_argument('--permutation', default=False, action="store_true")
    parser.add_argument('--randompermutation', default=False, action="store_true")
    parser.add_argument('--magwarp', default=False, action="store_true")
    parser.add_argument('--timewarp', default=False, action="store_true")
    parser.add_argument('--windowslice', default=False, action="store_true")
    parser.add_argument('--windowwarp', default=False, action="store_true")
    parser.add_argument('--rotation', default=False, action="store_true")
    parser.add_argument('--spawner', default=False, action="store_true")
    parser.add_argument('--dtwwarp', default=False, action="store_true")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true")
    parser.add_argument('--wdba', default=False, action="store_true")
    parser.add_argument('--discdtw', default=False, action="store_true")
    parser.add_argument('--discsdtw', default=False, action="store_true")
    parser.add_argument('--extra_tag', type=str, default="")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16)

    # Patch-based conv models (ModernTCN / TSLANet)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--patch_stride', type=int, default=8)
    parser.add_argument('--head_dropout', type=float, default=0.0)

    # FITS
    parser.add_argument('--cut_freq', type=int, default=0)

    # WFTNet
    parser.add_argument('--wavelet_scale', type=int, default=4)
    parser.add_argument('--period_coeff', type=float, default=0.5)

    # AdaWaveNet
    parser.add_argument('--lifting_kernel_size', type=int, default=7)
    parser.add_argument('--lifting_levels', type=int, default=1)
    parser.add_argument('--regu_details', type=float, default=0.01)
    parser.add_argument('--regu_approx', type=float, default=0.01)
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--sr_ratio', type=int, default=10)
    parser.add_argument('--output_attention', action='store_true')

    # GCN
    parser.add_argument('--node_dim', type=int, default=10)
    parser.add_argument('--gcn_depth', type=int, default=2)
    parser.add_argument('--gcn_dropout', type=float, default=0.3)
    parser.add_argument('--propalpha', type=float, default=0.3)
    parser.add_argument('--conv_channel', type=int, default=32)
    parser.add_argument('--skip_channel', type=int, default=32)

    parser.add_argument('--individual', action='store_true', default=False)

    # TimeFilter
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1)

    return parser


# ---------------------------------------------------------------------------
# Constraint helpers (keep the categorical spaces static; enforce constraints
# by post-processing so Optuna sees a consistent search space every trial).
# ---------------------------------------------------------------------------
def fix_heads(d_model, n_heads, options):
    """Return the largest head count in `options` that divides d_model
    and is <= the suggested n_heads (defensive; the configured sets already
    all divide their d_model options)."""
    if d_model % n_heads == 0:
        return n_heads
    cands = [h for h in sorted(options) if d_model % h == 0 and h <= n_heads]
    return cands[-1] if cands else 1


def dff_ge_dmodel(d_ff, d_model):
    """FFN width should be >= model width."""
    return max(d_ff, d_model)


# ---------------------------------------------------------------------------
# Per-model FIXED overrides (non-default constants for this task) and the
# per-model search-space suggestion functions. Each suggester returns a dict
# of {arg_name: value} for the *searched* dimensions only.
# ---------------------------------------------------------------------------
MODEL_FIXED = {
    # seq_len=66 constraint: only one down-sampling level halves cleanly.
    'TimeMixer': {'down_sampling_layers': 1, 'down_sampling_window': 2,
                  'down_sampling_method': 'avg'},
    # univariate -> single token, attention is ~a no-op; don't waste a BO dim.
    'iTransformer': {'n_heads': 4},
    # seq_len=66 constraint + univariate (n_clusters clamped to enc_in=1).
    'AdaWaveNet': {'lifting_levels': 1, 'n_clusters': 1},
    # DLinear / FITS: single channel -> per-channel head is identical.
    'DLinear': {},   # individual stays False (argparse default)
    'FITS': {},      # individual stays False (argparse default)
}


def suggest_DLinear(trial):
    return {
        'moving_avg': trial.suggest_categorical('moving_avg', [5, 7, 13, 25, 49]),  # odd
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 5e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_PatchTST(trial):
    d_model = trial.suggest_categorical('d_model', [16, 32, 64, 128])
    n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
    d_ff = trial.suggest_categorical('d_ff', [64, 128, 256])
    return {
        'd_model': d_model,
        'n_heads': fix_heads(d_model, n_heads, [2, 4, 8]),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),
        'd_ff': dff_ge_dmodel(d_ff, d_model),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_iTransformer(trial):
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    d_ff = trial.suggest_categorical('d_ff', [64, 128, 256])
    return {
        'd_model': d_model,
        'd_ff': dff_ge_dmodel(d_ff, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_TimesNet(trial):
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    d_ff = trial.suggest_categorical('d_ff', [32, 64, 128])
    return {
        'd_model': d_model,
        'd_ff': dff_ge_dmodel(d_ff, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),
        'top_k': trial.suggest_categorical('top_k', [2, 3, 5]),
        'num_kernels': trial.suggest_categorical('num_kernels', [4, 6]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_TimeMixer(trial):
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    d_ff = trial.suggest_categorical('d_ff', [32, 64, 128])
    return {
        'd_model': d_model,
        'd_ff': dff_ge_dmodel(d_ff, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3, 4, 5]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 3e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_MSGNet(trial):
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    d_ff = trial.suggest_categorical('d_ff', [32, 64, 128])
    return {
        'd_model': d_model,
        'd_ff': dff_ge_dmodel(d_ff, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),
        'top_k': trial.suggest_categorical('top_k', [2, 3, 5]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_ModernTCN(trial):
    patch_size = trial.suggest_categorical('patch_size', [8, 16])
    patch_stride = trial.suggest_categorical('patch_stride', [4, 8])
    return {
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
        'patch_size': patch_size,
        'patch_stride': min(patch_stride, patch_size),   # stride <= size
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'head_dropout': trial.suggest_float('head_dropout', 0.0, 0.3),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_FITS(trial):
    # cut_freq <= rFFT bins (seq_len//2 + 1); FITS clamps internally, but we
    # keep it well within range. `individual` stays False (single channel).
    return {
        'cut_freq': trial.suggest_int('cut_freq', 2, 33),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_TSLANet(trial):
    return {
        'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),   # emb_dim
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),     # depth
        'patch_size': trial.suggest_categorical('patch_size', [8, 16, 24, 32]),  # <=66
        'dropout': trial.suggest_float('dropout', 0.1, 0.6),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_WFTNet(trial):
    d_model = trial.suggest_categorical('d_model', [16, 32, 64])
    d_ff = trial.suggest_categorical('d_ff', [32, 64])
    return {
        'd_model': d_model,
        'd_ff': dff_ge_dmodel(d_ff, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2]),        # each layer = 1 CWT
        'top_k': trial.suggest_categorical('top_k', [0, 2, 3, 5]),        # 0 = wavelet-only
        'num_kernels': trial.suggest_categorical('num_kernels', [4, 6]),
        'wavelet_scale': trial.suggest_categorical('wavelet_scale', [2, 4, 6]),
        'period_coeff': trial.suggest_float('period_coeff', 0.1, 0.9),
        'dropout': trial.suggest_float('dropout', 0.0, 0.2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


def suggest_AdaWaveNet(trial):
    d_model = trial.suggest_categorical('d_model', [32, 64, 128, 256])
    d_ff = trial.suggest_categorical('d_ff', [32, 64, 128, 256])
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    return {
        'lifting_kernel_size': trial.suggest_categorical('lifting_kernel_size', [3, 5, 7]),
        'd_model': d_model,
        'd_ff': dff_ge_dmodel(d_ff, d_model),
        'e_layers': trial.suggest_categorical('e_layers', [1, 2, 3]),
        'n_heads': fix_heads(d_model, n_heads, [4, 8]),
        'regu_details': trial.suggest_categorical('regu_details', [0.0, 0.01, 0.1]),
        'regu_approx': trial.suggest_categorical('regu_approx', [0.0, 0.01, 0.1]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.2),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', BATCH_SIZES),
    }


SUGGESTERS = {
    'DLinear': suggest_DLinear,
    'PatchTST': suggest_PatchTST,
    'iTransformer': suggest_iTransformer,
    'TimesNet': suggest_TimesNet,
    'TimeMixer': suggest_TimeMixer,
    'MSGNet': suggest_MSGNet,
    'ModernTCN': suggest_ModernTCN,
    'FITS': suggest_FITS,
    'TSLANet': suggest_TSLANet,
    'WFTNet': suggest_WFTNet,
    'AdaWaveNet': suggest_AdaWaveNet,
}
ALL_MODELS = list(SUGGESTERS.keys())


# ---------------------------------------------------------------------------
# argv / args construction
# ---------------------------------------------------------------------------
def dict_to_argv(overrides):
    argv = []
    for key, val in overrides.items():
        if isinstance(val, bool):
            if val:
                argv.append(f'--{key}')
        else:
            argv += [f'--{key}', str(val)]
    return argv


def common_argv(cfg, model):
    """Fixed RV / aggregate-mean task config shared by every trial."""
    return [
        '--task_name', 'long_term_forecast', '--is_training', '1',
        '--model', model, '--model_id', f'RV_tune_{model}',
        '--data', 'custom', '--root_path', cfg.root_path,
        '--data_path', cfg.data_path, '--features', 'S', '--target', cfg.target,
        '--freq', 'd',
        '--seq_len', str(cfg.seq_len), '--label_len', str(cfg.label_len),
        '--pred_len', str(cfg.pred_len), '--aggregate_mean',
        '--enc_in', '1', '--dec_in', '1', '--c_out', '1',
        '--train_epochs', str(cfg.train_epochs), '--patience', str(cfg.patience),
        '--lradj', cfg.lradj, '--num_workers', '0', '--itr', '1',
        '--des', 'tune', '--checkpoints', cfg.checkpoints,
    ]


def finalize_device(args, device):
    import torch
    if device == 'cpu':
        args.use_gpu = False
    elif device == 'cuda':
        args.use_gpu = torch.cuda.is_available()
        args.gpu_type = 'cuda'
    else:  # auto
        args.use_gpu = torch.cuda.is_available()
        args.gpu_type = 'cuda' if args.use_gpu else args.gpu_type
    args.use_multi_gpu = False
    if args.use_gpu and args.gpu_type == 'cuda':
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')
    return args


def set_seed(seed):
    import torch
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def objective(trial, model, cfg, parser):
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    import torch

    # 1. sample the search space (with constraints already applied) and merge
    #    with the per-model fixed constants.
    searched = SUGGESTERS[model](trial)
    effective = {**MODEL_FIXED.get(model, {}), **searched}
    trial.set_user_attr('effective_config', effective)

    # 2. build the exact Namespace run.py would produce.
    argv = common_argv(cfg, model) + dict_to_argv(effective)
    args = parser.parse_args(argv)
    finalize_device(args, cfg.device)

    # 3. deterministic init/training per config (fair across trials).
    set_seed(cfg.seed)

    setting = f'tune_{model}_pl{cfg.pred_len}_t{trial.number}'
    ckpt_dir = os.path.join(args.checkpoints, setting)
    exp = None
    try:
        exp = Exp_Long_Term_Forecast(args)
        exp.train(setting)  # early-stops on val loss; reloads best checkpoint
        # validation aggregated MSE of the best (reloaded) checkpoint == the
        # early-stopping minimum (deterministic in eval mode).
        val_data, val_loader = exp._get_data(flag='val')
        val_mse = float(exp.vali(val_data, val_loader, exp._select_criterion()))
    except Exception as exc:  # bad/degenerate config -> prune, keep study alive
        print(f'[trial {trial.number}] {model} FAILED: {exc}')
        traceback.print_exc()
        raise optuna.TrialPruned() from exc
    finally:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        del exp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not math.isfinite(val_mse):
        raise optuna.TrialPruned()

    trial.set_user_attr('val_mse', val_mse)
    return val_mse


# ---------------------------------------------------------------------------
# Study runner + result serialization
# ---------------------------------------------------------------------------
def build_run_command(cfg, model, effective):
    """A copy/paste `python run.py` command reproducing the best config."""
    device_flag = [] if cfg.device != 'cpu' else ['--no_use_gpu']
    argv = [
        'python', '-u', 'run.py',
        '--task_name', 'long_term_forecast', '--is_training', '1',
        '--model', model, '--model_id', f'RV_best_{model}',
        '--data', 'custom', '--root_path', cfg.root_path,
        '--data_path', cfg.data_path, '--features', 'S', '--target', cfg.target,
        '--freq', 'd',
        '--seq_len', str(cfg.seq_len), '--label_len', str(cfg.label_len),
        '--pred_len', str(cfg.pred_len), '--aggregate_mean',
        '--enc_in', '1', '--dec_in', '1', '--c_out', '1',
        '--train_epochs', str(cfg.train_epochs), '--patience', str(cfg.patience),
        '--num_workers', '0', '--itr', '1', '--des', 'best',
    ] + dict_to_argv(effective) + device_flag
    return ' '.join(argv)


def run_study(model, cfg, parser):
    sampler = optuna.samplers.TPESampler(
        seed=cfg.seed,
        multivariate=True,      # joint (Bayesian) modelling of the search space
        group=True,
        n_startup_trials=min(10, max(1, cfg.n_trials // 4)),
    )
    storage = None
    if cfg.storage:
        storage = f'sqlite:///{cfg.storage}'
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name=f'RV_{model}_pl{cfg.pred_len}',
        storage=storage,
        load_if_exists=bool(storage),
    )
    t0 = time.time()
    study.optimize(
        lambda t: objective(t, model, cfg, parser),
        n_trials=cfg.n_trials,
        timeout=cfg.timeout,
        gc_after_trial=True,
    )
    elapsed = time.time() - t0

    best = study.best_trial
    effective = best.user_attrs.get('effective_config', best.params)
    result = {
        'model': model,
        'objective': 'validation_aggregated_MSE',
        'best_value': study.best_value,
        'best_trial_number': best.number,
        'searched_params': best.params,
        'effective_config': effective,
        'pred_len': cfg.pred_len,
        'seq_len': cfg.seq_len,
        'n_trials_requested': cfg.n_trials,
        'n_trials_completed': len([t for t in study.trials
                                   if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_trials_pruned': len([t for t in study.trials
                                if t.state == optuna.trial.TrialState.PRUNED]),
        'elapsed_sec': round(elapsed, 1),
        'run_command': build_run_command(cfg, model, effective),
    }
    return study, result


def save_result(result, out_dir):
    path = os.path.join(out_dir, f'{result["model"]}_best.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    # per-trial CSV for inspection
    return path


def dump_trials_csv(study, model, out_dir):
    try:
        df = study.trials_dataframe()
        df.to_csv(os.path.join(out_dir, f'{model}_trials.csv'), index=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_cli():
    p = argparse.ArgumentParser(
        description='Optuna (TPE / Bayesian) tuning for the 11 RV models.')
    p.add_argument('--models', type=str, default='all',
                   help="comma-separated subset or 'all' "
                        f"(options: {', '.join(ALL_MODELS)})")
    p.add_argument('--n_trials', type=int, default=40, help='BO trials per model')
    p.add_argument('--timeout', type=int, default=None,
                   help='optional per-model wall-clock limit (seconds)')
    p.add_argument('--pred_len', type=int, default=5,
                   help='aggregation horizon H (mean over next H days; 5=weekly, 22=monthly)')
    p.add_argument('--seq_len', type=int, default=66,
                   help='input length (66 default; use 64 to free TimeMixer/AdaWaveNet dims)')
    p.add_argument('--label_len', type=int, default=-1,
                   help='decoder start token length (ignored by all 11; default seq_len//2)')
    p.add_argument('--train_epochs', type=int, default=40, help='fixed (not a BO dim)')
    p.add_argument('--patience', type=int, default=8, help='fixed early-stopping patience')
    p.add_argument('--lradj', type=str, default='type1')
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--root_path', type=str, default='./data/')
    p.add_argument('--data_path', type=str, default='realized_volatility.csv')
    p.add_argument('--target', type=str, default='ln_RV')
    p.add_argument('--checkpoints', type=str, default='./checkpoints/tuning_ckpt/')
    p.add_argument('--out_dir', type=str, default='./tuning/results/')
    p.add_argument('--storage', type=str, default=None,
                   help='optional sqlite path to persist/resume studies '
                        '(e.g. tuning/results/rv_tuning.db)')
    cfg = p.parse_args()
    if cfg.label_len < 0:
        cfg.label_len = cfg.seq_len // 2
    return cfg


def main():
    cfg = parse_cli()
    optuna.logging.set_verbosity(optuna.logging.INFO)

    if cfg.models.strip().lower() == 'all':
        models = ALL_MODELS
    else:
        models = [m.strip() for m in cfg.models.split(',') if m.strip()]
        unknown = [m for m in models if m not in SUGGESTERS]
        if unknown:
            raise SystemExit(f'Unknown model(s): {unknown}. Options: {ALL_MODELS}')

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.checkpoints, exist_ok=True)
    parser = make_run_parser()

    print('=' * 78)
    print(f'Optuna TPE tuning | models={models} | n_trials={cfg.n_trials} '
          f'| pred_len={cfg.pred_len} | seq_len={cfg.seq_len} | device={cfg.device}')
    print('=' * 78)

    summary = []
    for model in models:
        print(f'\n########## Tuning {model} ##########')
        study, result = run_study(model, cfg, parser)
        save_result(result, cfg.out_dir)
        dump_trials_csv(study, model, cfg.out_dir)
        summary.append({'model': model,
                        'best_val_mse': result['best_value'],
                        'best_config': result['effective_config']})
        print(f'>>> {model}: best val MSE = {result["best_value"]:.6e}')
        print(f'    reproduce: {result["run_command"]}')

    summary.sort(key=lambda r: r['best_val_mse'])
    with open(os.path.join(cfg.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('\n' + '=' * 78)
    print('SUMMARY (best validation aggregated MSE, ascending)')
    print('=' * 78)
    for r in summary:
        print(f'  {r["model"]:<14} {r["best_val_mse"]:.6e}')
    print(f'\nResults written to {cfg.out_dir}')


if __name__ == '__main__':
    main()
