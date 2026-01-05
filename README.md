# Continuous-time Causal Discovery via Latent SDE and Control Energy Network

This repository implements the Causal Latent SDE framework for recovering causal graphs from continuous-time data. It combines latent-variable SDE modelling with control-energy regularisation to enforce sparse, directed dynamics and supports dense, sparse, and irregular observation regimes.

## Folder Overview

- `causal_sde.py` – main training module (encoder/decoder, structural drift and diagonal diffusion, control policy, KL and sparsity penalties) with Euler–Maruyama integration.
- `run_causal_sde.py` – baseline experiment for regularly sampled trajectories; logs MCC, SHD, F1, TPR, FDR per seed.
- `run_causal_sde_sparse.py`, `run_causal_sde_irregular.py` – variants tailored for sparse or irregular observations.
- `run_lorenz_scotch.py`, `run_glycolysis_scotch.py` – SCOTCH-based synthetic benchmarks (Lorenz96 and yeast glycolysis) that can either generate trajectories on-the-fly or load pre-generated datasets from `scotch/data/*_processed`.
- `datasets/` – benchmark data folders (Dream3/4, Glycolytic, NetSim) used by the original Causal-SDE scripts.
- `models/` – PyTorch encoder, decoder, structural SDE drift/diffusion modules, and control policy.
- `utils/` – simulation, evaluation metrics, locally-connected layers, optimisation wrappers, and visualisation helpers.
- `results/` – plain-text experiment logs, including SCOTCH runs such as [results/lorenz96_scotch_settings.txt](results/lorenz96_scotch_settings.txt).

## Getting Started

1. **Environment**: Install PyTorch, NumPy, SciPy, and tqdm. For SCOTCH generation, install `torchsde` (optional; scripts fall back to Euler–Maruyama). GPU determinism is enabled when a seed is provided.
2. **Standard Causal-SDE runs** (regular/sparse/irregular sampling):
   ```bash
   python run_causal_sde.py
   python run_causal_sde_sparse.py
   python run_causal_sde_irregular.py
   ```
   Adjust hyperparameters inside each script (learning rate, drift/diffusion widths, training steps, sampling mode).
3. **SCOTCH benchmarks** (Lorenz96 or glycolysis):
   ```bash
   # On-the-fly generation (default)
   python run_lorenz_scotch.py
   python run_glycolysis_scotch.py

   # Use pre-generated datasets
   # Inside the script: set config.use_preprocessed_data = True
   # Expected files live under scotch/data/lorenz96_processed or scotch/data/yeast_processed.
   ```
   Key toggles in the scripts: `config.experiment_name` (`lorenz96` or `glycolysis`), `config.scotch_missing_prob` (missingness), `config.scotch_num_subsamp` (subsampled grid), and `config.use_preprocessed_data` (pre-generated vs. simulated trajectories).
4. **Inspect outputs**: Metrics are appended to the corresponding file in `results/` (e.g., [results/lorenz96_scotch_settings.txt](results/lorenz96_scotch_settings.txt) or `results/causal_sde_results.txt`). Each line records seed, method, sampling mode, training steps, MCC, AUROC, SHD, F1, TPR, FDR, and wall-clock time.

## Recent SCOTCH Results

- Lorenz96, irregular sampling with diagonal diffusion: AUROC means of 0.6633 (fraction 0.4, 4000 samples) and 0.7358 (fraction 0.7, 7000 samples) across seeds 42–46; see [results/lorenz96_scotch_settings.txt](results/lorenz96_scotch_settings.txt).

## Reproducibility

All runner scripts seed Python `random`, NumPy, and PyTorch (CPU/GPU) and enable deterministic CuDNN to make repeated runs comparable for the same seeds.

## Extending the Framework

- Adjust `models/SDEmlp.py` for alternative drift/diffusion architectures or structural constraints.
- Experiment with control strategies through `models/ControlPolicy.py` or by modifying the sparsity and control-energy penalties inside `causal_sde.py`.
