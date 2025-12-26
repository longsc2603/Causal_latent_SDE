import argparse
import random
import torch
from time import time

import numpy as np

from causal_sde import CausalSDE
from utils.evaluation import MetricsDAG
from utils.simulator import simulate_latent_lorenz

parser = argparse.ArgumentParser(description='Configuration for Causal-SDE Irregular')
config = parser.parse_args(args=[])

# Model architecture parameters
config.n_v = 100
config.n_z = 30
config.n_samples = 500
config.delta_t = 0.1
config.fraction = 0.3
config.dims = [config.n_v, 1]
config.z_dims = [config.n_z, 10, 1]

# Regularization parameters
config.lambda1 = 0.1   # M-regularization (encoder-decoder coupling)
config.lambda2 = 0.05  # L2 regularization for drift
config.lambda3 = 0.002 # Proximal for encoder/decoder
config.lambda4 = 0.01  # Proximal for SDE drift
config.lambda5 = 0.01  # L2 regularization for diffusion
config.gamma = 0.5
config.beta_control = 0.01
config.control_context = 'latent'  # Options: 'none', 'observation', 'latent'
config.control_hidden = 16
config.control_layers = 2


# Training parameters
config.bias = True
config.irregular = 'irregular'  # 'frequent', 'sparse', or 'irregular' sampling
config.lr = 0.005
config.n_auto_steps = 10000
config.pretrain_steps = 2000
config.lasso_type = 'AGL'
config.w_threshold = 0.1

# Device configuration
config.device_type = 'gpu'
config.device_ids = 0

# SDE-specific parameters
config.noise_type = 'general'  # 'scalar', 'diagonal', or 'general'
config.sde_type = 'ito'

print("="*80)
print("Causal-SDE (Irregular): Causal Discovery with Neural SDEs on irregular samples")
print("="*80)
print(f"Noise Type: {config.noise_type}")
print(f"SDE Type: {config.sde_type}")
print("="*80)

# Experiments with different system sizes
# (n_observed, n_latent, learning_rate, lambda3, lambda4)
experiments = [
    (100, 30, 0.005, 0.002, 0.01)
]

results_file = 'results/causal_sde_irregular_results.txt'

for dx, dz, lr, l3, l4 in experiments:
    config.n_v = dx
    config.n_z = dz
    config.dims = [dx, 1]
    config.z_dims = [dz, 10, 1]
    config.lr = lr
    config.lambda3 = l3
    config.lambda4 = l4

    print(f"\n{'='*80}")
    print(f"Running experiment: n_v={dx}, n_z={dz}, lr={lr}")
    print(f"{'='*80}")

    for k in range(5):
        print(f"\nRun {k+1}/5...")
        k += 42
        np.random.seed(k)
        random.seed(k)
        torch.manual_seed(k)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(k)
        config.seed = k

        # Generate latent Lorenz synthetic data
        T = int(config.n_samples / config.fraction)
        data, Z, GC, B, Z_B = simulate_latent_lorenz(
            n_v=config.n_v,
            n_z=config.n_z,
            weight_range=(0.5, 2),
            T=T,
            delta_t=config.delta_t,
            method='linear',
            seed=k
        )

        # Irregular sampling: pick a subset of time indices
        indices = np.sort(np.random.choice(range(T), config.n_samples, replace=False))
        times = np.linspace(0, T, T).astype(int)
        times = times[indices]
        data = data[indices, :]
        Z = Z[indices, :]

        # Initialize and train SDE model
        causal_sde = CausalSDE(config)
        begin_time = time()
        causal_sde.learn(data, times)
        end_time = time()

        # Evaluate latent recovery
        W_corrs, B_corrs = causal_sde.get_MCC(data, Z)

        # Get causal adjacency estimates
        GC_est, W_xz, W_z, W_zx = causal_sde.get_adj()

        # Evaluate causal discovery in latent space
        B_est = np.matmul(np.matmul(B_corrs, W_z), B_corrs.T)
        met = MetricsDAG(B_est, B)

        # Save results (include TPR, FDR, F1)
        with open(results_file, 'a') as f:
            f.write(f'seed:{config.seed},')
            f.write(f'method:Causal-SDE-irregular-{config.noise_type}-{dx}-{dz},')
            f.write(f'sampling_mode:{config.irregular},')
            f.write(f'training_steps:{config.n_auto_steps},')
            f.write(f'MCC:{W_corrs:.4f},')
            f.write(f'SHD:{met.metrics["shd"]},')
            f.write(f'F1:{met.metrics["F1"]:.4f},')
            f.write(f'TPR:{met.metrics["tpr"]:.4f},')
            f.write(f'FDR:{met.metrics["fdr"]:.4f},')
            f.write(f'time:{end_time-begin_time:.2f}s\n')

        print(f"  MCC: {W_corrs:.4f}, SHD: {met.metrics['shd']}, F1: {met.metrics['F1']:.4f}")
        print(f"  TPR: {met.metrics['tpr']:.4f}, FDR: {met.metrics['fdr']:.4f}")
        print(f"  Training time: {end_time-begin_time:.2f}s")

print("\n" + "="*80)
print("All experiments completed!")
print(f"Results saved to: {results_file}")
print("="*80)
