import argparse
import random
import torch
from time import time

import numpy as np

from causal_sde import CausalSDE
from utils.evaluation import MetricsDAG, compute_auroc
from utils.simulator import simulate_latent_lorenz

parser = argparse.ArgumentParser(description='Configuration for Causal-SDE')
config = parser.parse_args(args=[])

if __name__ == "__main__":
    # Model architecture parameters
    config.n_v = 100
    config.n_z = 30
    config.n_samples = 2000
    config.obs_delta_t = 0.1
    config.delta_t = 0.1
    config.fraction = 1
    config.dims = [config.n_v, 1]
    config.z_dims = [config.n_z, 10, 1]

    # Regularization parameters
    config.lambda1 = 0.05   # M-regularization (encoder-decoder coupling)
    config.lambda2 = 0.05  # L2 regularization for drift
    config.lambda3 = 0.002 # Proximal for encoder/decoder
    config.lambda4 = 0.01  # Proximal for SDE drift
    config.lambda5 = 0.05  # L2 regularization for diffusion
    config.gamma = 0.5
    config.beta_control = 0.01
    config.control_context = 'latent'  # Options: 'none', 'observation', 'latent'
    config.control_hidden = 64
    config.control_layers = 2

    # Training parameters
    config.bias = True
    config.irregular = 'frequent'  # 'frequent', 'sparse', or 'irregular' sampling
    config.lr = 0.005
    config.train_steps = 30000
    config.pretrain_steps = 2000
    config.lasso_type = 'AGL'
    config.w_threshold = 0.1

    # Device configuration
    config.device_type = 'gpu'
    config.device_ids = 0

    # SDE-specific parameters
    config.noise_type = 'general'  # 'scalar', 'diagonal', or 'general'
    config.sde_type = 'ito'        # 'ito' or 'stratonovich'

    print("="*80)
    print("Causal-SDE: Causal Discovery with Neural Stochastic Differential Equations")
    print("="*80)
    print(f"Noise Type: {config.noise_type}")
    print(f"SDE Type: {config.sde_type}")
    print("="*80)

    results_file = 'results/causal_sde_results.txt'
        
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
        
        # Generate synthetic data
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
        
        # Sample observations
        indices = np.sort(np.random.choice(range(T), config.n_samples, replace=False))
        times = np.linspace(0, T, T)
        times = times[indices]
        data = data[indices, :]
        Z = Z[indices, :]
        
        # Initialize and train model
        causal_sde = CausalSDE(config)
        begin_time = time()
        causal_sde.learn(data, times, Z, B)
        end_time = time()
        
        # Evaluate latent variable recovery
        W_corrs, B_corrs = causal_sde.get_MCC(data, Z)
        
        # Get causal structure
        GC_est, W_xz, W_z, W_zx = causal_sde.get_adj()
        
        # Evaluate causal discovery in latent space
        B_est = np.matmul(np.matmul(B_corrs, W_z), B_corrs.T)
        
        # AUROC calculation
        auroc = compute_auroc(B, B_est, diagonal=True)

        met = MetricsDAG(B_est, B)
        
        # Save results (include TPR, FDR, F1)
        with open(results_file, 'a') as f:
            f.write(f'seed:{config.seed},')
            f.write(f'method:Causal-SDE-{config.noise_type}-{config.n_z}-{config.n_v},')
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
