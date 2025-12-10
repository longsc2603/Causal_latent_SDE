# Causal Latent SDE

This directory implements the Causal Latent Stochastic Differential Equation (SDE) framework for discovering causal structure from continuous-time data. The codebase combines latent-variable SDE modelling with control-based regularisation to recover directed acyclic graphs in the latent dynamics.

## Folder Overview

- `causal_sde.py` – main training module. Couples encoder/decoder networks, structural SDE drift, diffusion parameterisation, and control policy learning. Implements Euler–Maruyama integration, KL regularisation, and proximal updates for sparsity.
- `run_causal_sde.py` – reference experiment script for the frequent/sparse sampling regime; seeds all random number generators and logs evaluation metrics (MCC, SHD, F1, TPR, FDR).
- `run_causal_sde_sparse.py`, `run_causal_sde_irregular.py` – variants configured for sparse or irregular observation patterns.
- `generate_glycolytic_data.py` – synthetic data generator for the glycolytic oscillator benchmark.
- `datasets/` – local copies of benchmark datasets (Dream3, Dream4, Glycolytic, NetSim). Replace or augment these folders if you need updated data.
- `models/` – PyTorch modules for the encoder, decoder, SDE drift/diffusion (with adjacency masking), and control policy.
- `utils/` – supporting utilities for simulation, evaluation metrics, locally-connected layers, optimisation wrappers, and visualisation helpers.
- `results/` – experiment logs (plain-text) capturing evaluation statistics for each run.

## Getting Started

1. **Environment**: Install dependencies such as PyTorch, NumPy, SciPy, and tqdm. For GPU determinism, the scripts configure CuDNN to deterministic mode when a seed is provided.
2. **Run an experiment**:
   ```bash
   python Causal_latent_SDE/run_causal_sde.py
   ```
   Adjust hyperparameters inside the script or extend it with CLI arguments as needed.
3. **Inspect outputs**: After training, review `results/Causal_sde_results.txt` (and corresponding files for sparse/irregular runs) to compare MCC, SHD, and other metrics across seeds.

## Reproducibility

The runner scripts seed Python `random`, NumPy, and PyTorch (CPU/GPU) at the beginning of each trial. They also enable deterministic CuDNN behaviour, allowing repeated runs to reproduce results for the same seeds.

## Extending the Framework

- Modify `models/SDEmlp.py` if you need custom drift or diffusion architectures, or alternative structural constraints.
- Update `utils/simulator.py` to plug in new generative processes or real datasets.
- Use the control policy in `models/ControlPolicy.py` to experiment with different steering strategies or context inputs.

For additional background, consult the accompanying research proposal or project documentation in the parent repository.
