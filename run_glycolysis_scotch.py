import argparse
import math
import random
import torch
from time import time
from pathlib import Path

import numpy as np

from causal_sde import CausalSDE
from utils.evaluation import MetricsDAG, compute_auroc

parser = argparse.ArgumentParser(description='Configuration for Causal-SDE Irregular')
config = parser.parse_args(args=[])

config.experiment_name = 'glycolysis' # 'lorenz96' or 'glycolysis'

# If True, load pre-generated datasets from scotch/data/*_processed.
# If False, generate trajectories on-the-fly when running this script.
config.use_preprocessed_data = False


def _device_str_from_config(device_type: str) -> str:
    if device_type == 'gpu' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def _float_to_filename_str(x: float) -> str:
    """Match the on-disk SCOTCH naming convention, e.g. 0.0, 0.3, 0.6."""
    return f"{float(x):.1f}"


def _load_scotch_timeseries_from_disk(config, seed_id: int):
    """Load one pre-generated SCOTCH dataset from scotch/data.

    The saved files were produced by scotch/dataset_generation/generate_and_save_data.py.

    Returns:
        data_full: np.ndarray of shape [T, d]
        Z_full: np.ndarray of shape [T, d] (here equals data_full)
        B: np.ndarray of shape [d, d] (binary adjacency)
        dt_base: float (base observation interval corresponding to one index step)
    """
    subf = "norm" if config.scotch_normalize else "unnorm"
    missing_s = _float_to_filename_str(config.scotch_missing_prob)

    if config.experiment_name == "lorenz96":
        processed_dir = Path("scotch") / "data" / "lorenz96_processed" / str(config.n_v) / subf
    elif config.experiment_name == "glycolysis":
        processed_dir = Path("scotch") / "data" / "yeast_processed" / str(config.n_v) / subf
    else:
        raise ValueError(f"Unknown experiment_name: {config.experiment_name}")

    data_path = processed_dir / f"data_{config.scotch_num_subsamp}_{missing_s}_{seed_id}.pt"
    times_path = processed_dir / f"times_{config.scotch_num_subsamp}_{missing_s}_{seed_id}.pt"
    graph_path = processed_dir / "true_graph.pt"

    if not data_path.exists() or not times_path.exists() or not graph_path.exists():
        raise FileNotFoundError(
            "Missing pre-generated SCOTCH dataset files. Expected:\n"
            f"  {data_path}\n  {times_path}\n  {graph_path}\n"
            "Check scotch/data/*_processed/*/(norm|unnorm) and your config scotch_* settings."
        )

    # Keep loads on CPU; CausalSDE.learn expects numpy arrays anyway.
    training_data = torch.load(data_path, map_location="cpu")
    ts = torch.load(times_path, map_location="cpu")
    true_graph = torch.load(graph_path, map_location="cpu")

    if not torch.is_tensor(training_data):
        raise TypeError(f"Expected tensor in {data_path}, got {type(training_data)}")
    if not torch.is_tensor(ts):
        raise TypeError(f"Expected tensor in {times_path}, got {type(ts)}")
    if not torch.is_tensor(true_graph):
        raise TypeError(f"Expected tensor in {graph_path}, got {type(true_graph)}")

    # training_data: (train_size, num_time_points_after_missing, d)
    if training_data.ndim != 3:
        raise ValueError(f"Expected training_data to be 3D, got shape {tuple(training_data.shape)}")
    d = int(training_data.shape[-1])
    if d != int(config.n_v):
        raise ValueError(f"Loaded data has d={d}, but config.n_v={config.n_v}")

    data_full = training_data.reshape(-1, d).detach().cpu().numpy()
    Z_full = data_full.copy()
    B = true_graph.detach().cpu().numpy().astype(int)

    # Base dt for one index step on the subsampled regular grid.
    dt_base = float(config.scotch_t_max) / float(config.scotch_num_subsamp)

    # Sanity-check that times length matches the trajectory time dimension.
    if int(ts.shape[0]) != int(training_data.shape[1]):
        raise ValueError(
            f"times length ({int(ts.shape[0])}) != data time dimension ({int(training_data.shape[1])})."
        )

    return data_full, Z_full, B, dt_base


def _euler_maruyama_simulate(sde, z0: torch.Tensor, ts: torch.Tensor, dt: float) -> torch.Tensor:
    """Euler-Maruyama simulation.

    Returns tensor shaped like torchsde output: (time, batch, state).
    """
    if ts.ndim != 1:
        raise ValueError("ts must be 1D")
    if z0.ndim != 2:
        raise ValueError("z0 must be (batch, state)")

    y = z0
    ys = [y]
    for i in range(1, ts.shape[0]):
        t0 = ts[i - 1]
        t1 = ts[i]
        interval = float((t1 - t0).item())
        n_steps = max(1, int(math.ceil(interval / float(dt))))
        dt_step = interval / float(n_steps)
        t = float(t0.item())

        for _ in range(n_steps):
            drift = sde.f(t, y)
            diffusion = sde.g(t, y)
            dW = torch.randn_like(y) * math.sqrt(dt_step)
            y = y + drift * dt_step + diffusion * dW
            t += dt_step

        ys.append(y)

    return torch.stack(ys, dim=0)


def _simulate_scotch_sde(sde, z0: torch.Tensor, num_time_points: int, t_max: float, dt: float, device: str):
    """Try torchsde.sdeint, fall back to Euler–Maruyama if torchsde isn't available."""
    ts = torch.linspace(0, t_max, num_time_points, device=device)

    try:
        import torchsde

        bm = torchsde.BrownianInterval(
            t0=0.0,
            t1=t_max,
            size=z0.shape,
            levy_area_approximation="space-time",
            device=device,
        )
        zs = torchsde.sdeint(sde, z0, ts, bm=bm, dt=dt, method="euler")
        return ts, zs  # (time, batch, state)
    except (ModuleNotFoundError, ImportError, RecursionError):
        zs = _euler_maruyama_simulate(sde=sde, z0=z0, ts=ts, dt=dt)
        return ts, zs


def _generate_scotch_timeseries(config):
    """Generate one multivariate time series + ground-truth graph using SCOTCH generators.

    Returns:
        data_full: np.ndarray of shape [T, d]
        Z_full: np.ndarray of shape [T, d] (here equals data_full)
        B: np.ndarray of shape [d, d] (binary adjacency)
        dt_base: float (base observation interval corresponding to one index step)
    """
    device = _device_str_from_config(config.device_type)

    if config.experiment_name == 'lorenz96':
        from scotch.dataset_generation.example_sdes.lorenz_sde import Lorenz96SDE

        d = config.n_v
        z0 = torch.randn(size=(config.scotch_train_size, d), device=device)
        sde = Lorenz96SDE(F=config.scotch_forcing, noise_scale=config.scotch_noise_scale)
        ts, zs = _simulate_scotch_sde(
            sde=sde,
            z0=z0,
            num_time_points=config.scotch_num_time_points,
            t_max=config.scotch_t_max,
            dt=config.scotch_solver_dt,
            device=device,
        )
        trajectories = zs.permute(1, 0, 2)  # (batch, time, state)
        true_graph = Lorenz96SDE.graph(d)
    elif config.experiment_name == 'glycolysis':
        from scotch.dataset_generation.example_sdes.yeast_glycolysis import YeastGlycolysisSDE

        d = config.n_v
        variable_ranges = torch.tensor(
            [
                [0.15, 1.60],
                [0.19, 2.16],
                [0.04, 0.20],
                [0.10, 0.35],
                [0.08, 0.30],
                [0.14, 2.67],
                [0.05, 0.10],
            ],
            device=device,
        )
        unif_samples = torch.rand(size=(config.scotch_train_size, d), device=device)
        z0 = variable_ranges[:, 0] + (variable_ranges[:, 1] - variable_ranges[:, 0]) * unif_samples
        sde = YeastGlycolysisSDE(noise_scale=config.scotch_noise_scale)
        ts, zs = _simulate_scotch_sde(
            sde=sde,
            z0=z0,
            num_time_points=config.scotch_num_time_points,
            t_max=config.scotch_t_max,
            dt=config.scotch_solver_dt,
            device=device,
        )
        trajectories = zs.permute(1, 0, 2)  # (batch, time, state)
        true_graph = YeastGlycolysisSDE.graph()
    else:
        raise ValueError(f"Unknown experiment_name: {config.experiment_name}")

    # trajectories: [batch, T, d]
    if config.scotch_normalize:
        mean = trajectories.mean(dim=(0, 1), keepdim=True)
        std = trajectories.std(dim=(0, 1), keepdim=True)
        trajectories = (trajectories - mean) / std

    # Optional subsampling (regular grid)
    if config.scotch_num_subsamp is not None and config.scotch_num_subsamp < trajectories.shape[1]:
        samp_every = max(1, int(trajectories.shape[1] // config.scotch_num_subsamp))
        ts = ts[::samp_every]
        trajectories = trajectories[:, ::samp_every, :]

    # Optional missing observations (irregular; drop entire timepoints across all variables)
    if config.scotch_missing_prob > 0:
        mask = torch.rand(trajectories.shape[1], device=trajectories.device) > config.scotch_missing_prob
        ts = ts[mask]
        trajectories = trajectories[:, mask, :]

    data_full = trajectories.reshape(-1, config.n_v).detach().cpu().numpy()
    Z_full = data_full.copy()

    B = true_graph.detach().cpu().numpy().astype(int)

    # Base dt for one index step on the (subsampled) regular grid.
    dt_base = float(config.scotch_t_max) / float(config.scotch_num_subsamp)

    return data_full, Z_full, B, dt_base


    config.n_v = 7
    config.n_z = 7
    config.n_samples = 10000
    config.scotch_normalize = True
    config.scotch_solver_dt = 0.005
    config.scotch_noise_scale = 0.01


if __name__ == '__main__':
    # SCOTCH generation settings
    config.scotch_train_size = 100
    config.scotch_num_time_points = 100
    config.scotch_num_subsamp = 100
    config.scotch_t_max = 100.0
    config.scotch_missing_prob = 0.0
    config.scotch_forcing = 10.0
    config.scotch_normalize = True
    config.scotch_solver_dt = 0.005
    config.scotch_noise_scale = 0.01

    config.n_v = 7
    config.n_z = 7
    config.n_samples = 10000
    config.fraction = 1.0  # Fraction of observed time points (for irregular sampling)

    # Our own settings
    config.dims = [config.n_v, 1]
    config.z_dims = [config.n_z, 10, 1]
    config.delta_t = 0.1  # Method solver step size

    # Regularization parameters
    config.lambda1 = 0.1   # M-regularization (encoder-decoder coupling)
    config.lambda2 = 0.05  # L2 regularization for drift
    config.lambda3 = 0.01 # Proximal for encoder/decoder
    config.lambda4 = 0.01  # Proximal for SDE drift
    config.lambda5 = 0.05  # L2 regularization for diffusion
    config.gamma = 0.5
    config.beta_control = 0.01
    config.control_context = 'latent'  # Options: 'none', 'observation', 'latent'
    config.control_hidden = 64
    config.control_layers = 2
    config.annealing_proximal = True


    # Training parameters
    config.bias = True
    config.irregular = 'irregular'  # 'frequent', 'sparse', or 'irregular' sampling
    config.lr = 0.003
    config.train_steps = 2500
    config.pretrain_steps = 3000
    config.lasso_type = 'AGL'
    config.w_threshold = 0.1

    # Device configuration
    config.device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
    config.device_ids = 0

    # SDE-specific parameters
    config.noise_type = 'diagonal'  # 'scalar', 'diagonal', or 'general'
    config.sde_type = 'ito'

    print("="*80)
    print("Causal-SDE (Irregular): Causal Discovery with Neural SDEs on irregular samples")
    print("="*80)
    print(f"Noise Type: {config.noise_type}")
    print(f"SDE Type: {config.sde_type}")
    print("="*80)

    results_file = 'results/glycolysis_scotch_settings.txt'


    # Experiments
    print(f"\n{'='*80}")
    print(f"Running experiment: n_v={config.n_v}, n_z={config.n_z}, lr={config.lr}")
    print(f"{'='*80}")

    num_runtimes = 5
    f1_scores = []
    tpr_scores = []
    fdr_scores = []
    auroc_scores = []
    for k in range(num_runtimes):
        print(f"\nRun {k+1}/{num_runtimes}...")
        k += 42
        np.random.seed(k)
        random.seed(k)
        torch.manual_seed(k)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(k)
        config.seed = k

        if config.use_preprocessed_data:
            # ---- Load pre-generated SCOTCH dataset (saved under scotch/data) ----
            # Saved datasets are indexed by seed in {0,1,2,3,4}; map current run seed onto that range.
            seed_id = int(config.seed) % 5
            data_full, Z_full, B, dt_base = _load_scotch_timeseries_from_disk(config, seed_id=seed_id)
        else:
            # ---- Generate SCOTCH trajectories on-the-fly ----
            data_full, Z_full, B, dt_base = _generate_scotch_timeseries(config)
        config.obs_delta_t = dt_base

        # ---- Subsample into an irregular observation pattern expected by CausalSDE.learn() ----
        T_full = data_full.shape[0]
        n_samples = min(int(config.n_samples), int(T_full))
        print('Number of time points in full data:', T_full)
        print('Number of observed time points:', n_samples)
        indices = np.sort(np.random.choice(np.arange(T_full), n_samples, replace=False))
        times = indices.astype(int)
        data = data_full[indices, :]
        Z = Z_full[indices, :]

        # Initialize and train SDE model
        causal_sde = CausalSDE(config)
        begin_time = time()
        causal_sde.learn(data, times, Z, B)
        end_time = time()

        # Evaluate latent recovery
        W_corrs, B_corrs = causal_sde.get_MCC(data, Z)

        # Get causal adjacency estimates
        GC_est, W_xz, W_z, W_zx = causal_sde.get_adj()

        # Evaluate causal discovery in latent space
        B_est = np.matmul(np.matmul(B_corrs, W_z), B_corrs.T)
        
        # AUROC calculation
        auroc = compute_auroc(B, B_est, diagonal=True)

        met = MetricsDAG(B_est, B)

        # Save results (include TPR, FDR, F1)
        with open(results_file, 'a') as f:
            f.write(f'seed:{config.seed},')
            f.write(f'method:Causal-SDE-irregular-{config.noise_type}-{config.n_v}-{config.n_z},')
            f.write(f'sampling_mode:{config.irregular},')
            f.write(f'training_steps:{config.train_steps},')
            f.write(f'MCC:{W_corrs:.4f},')
            f.write(f'AUROC:{auroc:.4f},')
            f.write(f'SHD:{met.metrics["shd"]},')
            f.write(f'F1:{met.metrics["F1"]:.4f},')
            f.write(f'TPR:{met.metrics["tpr"]:.4f},')
            f.write(f'FDR:{met.metrics["fdr"]:.4f},')
            f.write(f'time:{end_time-begin_time:.2f}s\n')

        print(f"  MCC: {W_corrs:.4f}, SHD: {met.metrics['shd']}, F1: {met.metrics['F1']:.4f}")
        print(f"  TPR: {met.metrics['tpr']:.4f}, FDR: {met.metrics['fdr']:.4f}, AUROC: {auroc:.4f}")
        print(f"  Training time: {end_time-begin_time:.2f}s")
        f1_scores.append(met.metrics['F1'])
        tpr_scores.append(met.metrics['tpr'])
        fdr_scores.append(met.metrics['fdr'])
        auroc_scores.append(auroc)

    # Calculate and print metric statistics
    print(f"\nMetrics Statistics ({num_runtimes} runs):")
    print(f"  F1:  Mean = {np.mean(f1_scores):.4f}, Std = {np.std(f1_scores):.4f}")
    print(f"  TPR: Mean = {np.mean(tpr_scores):.4f}, Std = {np.std(tpr_scores):.4f}")
    print(f"  FDR: Mean = {np.mean(fdr_scores):.4f}, Std = {np.std(fdr_scores):.4f}")
    print(f"  AUROC: Mean = {np.mean(auroc_scores):.4f}, Std = {np.std(auroc_scores):.4f}")


    print("\n" + "="*80)
    print("All experiments completed!")
    print(f"Results saved to: {results_file}")
    print("="*80)
