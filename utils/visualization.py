"""
Visualization utilities for comparing ODE and SDE dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_trajectory_comparison(data_clean, data_noisy, z_ode, z_sde, save_path='results/trajectory_comparison.png'):
    """
    Plot trajectories to visualize ODE vs SDE behavior
    
    Args:
        data_clean: Clean observed data
        data_noisy: Noisy observed data
        z_ode: Latent trajectories from ODE model
        z_sde: Latent trajectories from SDE model
    """
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot observed data
    ax1 = fig.add_subplot(gs[0, :])
    time_steps = np.arange(len(data_noisy))
    ax1.plot(time_steps[:200], data_clean[:200, 0], 'b-', linewidth=2, alpha=0.7, label='Clean signal')
    ax1.plot(time_steps[:200], data_noisy[:200, 0], 'r.', markersize=3, alpha=0.5, label='Noisy observations')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Observed Variable 1')
    ax1.set_title('Observed Data: Clean vs Noisy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot ODE latent trajectories
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(min(3, z_ode.shape[1])):
        ax2.plot(time_steps[:200], z_ode[:200, i], label=f'Latent {i+1}', linewidth=1.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Latent Variables')
    ax2.set_title('ODE Model: Latent Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot SDE latent trajectories
    ax3 = fig.add_subplot(gs[1, 1])
    for i in range(min(3, z_sde.shape[1])):
        ax3.plot(time_steps[:200], z_sde[:200, i], label=f'Latent {i+1}', linewidth=1.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Latent Variables')
    ax3.set_title('SDE Model: Latent Trajectories')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot phase space (2D projection)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(z_ode[:, 0], z_ode[:, 1], 'b-', linewidth=1, alpha=0.6)
    ax4.scatter(z_ode[0, 0], z_ode[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax4.scatter(z_ode[-1, 0], z_ode[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    ax4.set_xlabel('Latent Variable 1')
    ax4.set_ylabel('Latent Variable 2')
    ax4.set_title('ODE Model: Phase Space')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(z_sde[:, 0], z_sde[:, 1], 'r-', linewidth=1, alpha=0.6)
    ax5.scatter(z_sde[0, 0], z_sde[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax5.scatter(z_sde[-1, 0], z_sde[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    ax5.set_xlabel('Latent Variable 1')
    ax5.set_ylabel('Latent Variable 2')
    ax5.set_title('SDE Model: Phase Space')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('ODE vs SDE: Trajectory Comparison', fontsize=16, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory comparison plot saved to: {save_path}")
    plt.close()

def plot_performance_comparison(results, save_path='results/performance_comparison.png'):
    """
    Plot performance metrics across different noise levels
    
    Args:
        results: Dictionary with performance metrics from compare_ode_sde.py
    """
    noise_levels = sorted(results['ode'].keys())
    
    # Calculate means
    ode_mcc = [np.mean(results['ode'][nl]['mcc']) for nl in noise_levels]
    sde_mcc = [np.mean(results['sde'][nl]['mcc']) for nl in noise_levels]
    ode_shd = [np.mean(results['ode'][nl]['shd']) for nl in noise_levels]
    sde_shd = [np.mean(results['sde'][nl]['shd']) for nl in noise_levels]
    ode_f1 = [np.mean(results['ode'][nl]['f1']) for nl in noise_levels]
    sde_f1 = [np.mean(results['sde'][nl]['f1']) for nl in noise_levels]
    
    # Calculate std
    ode_mcc_std = [np.std(results['ode'][nl]['mcc']) for nl in noise_levels]
    sde_mcc_std = [np.std(results['sde'][nl]['mcc']) for nl in noise_levels]
    ode_shd_std = [np.std(results['ode'][nl]['shd']) for nl in noise_levels]
    sde_shd_std = [np.std(results['sde'][nl]['shd']) for nl in noise_levels]
    ode_f1_std = [np.std(results['ode'][nl]['f1']) for nl in noise_levels]
    sde_f1_std = [np.std(results['sde'][nl]['f1']) for nl in noise_levels]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MCC plot
    axes[0].errorbar(noise_levels, ode_mcc, yerr=ode_mcc_std, marker='o', 
                     label='ODE', linewidth=2, capsize=5, markersize=8)
    axes[0].errorbar(noise_levels, sde_mcc, yerr=sde_mcc_std, marker='s', 
                     label='SDE', linewidth=2, capsize=5, markersize=8)
    axes[0].set_xlabel('Noise Level', fontsize=12)
    axes[0].set_ylabel('MCC Score', fontsize=12)
    axes[0].set_title('Mean Correlation Coefficient', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # SHD plot
    axes[1].errorbar(noise_levels, ode_shd, yerr=ode_shd_std, marker='o', 
                     label='ODE', linewidth=2, capsize=5, markersize=8)
    axes[1].errorbar(noise_levels, sde_shd, yerr=sde_shd_std, marker='s', 
                     label='SDE', linewidth=2, capsize=5, markersize=8)
    axes[1].set_xlabel('Noise Level', fontsize=12)
    axes[1].set_ylabel('SHD (lower is better)', fontsize=12)
    axes[1].set_title('Structural Hamming Distance', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # F1 plot
    axes[2].errorbar(noise_levels, ode_f1, yerr=ode_f1_std, marker='o', 
                     label='ODE', linewidth=2, capsize=5, markersize=8)
    axes[2].errorbar(noise_levels, sde_f1, yerr=sde_f1_std, marker='s', 
                     label='SDE', linewidth=2, capsize=5, markersize=8)
    axes[2].set_xlabel('Noise Level', fontsize=12)
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('F1 Score', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('ODE vs SDE: Performance Comparison Across Noise Levels', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Performance comparison plot saved to: {save_path}")
    plt.close()

def plot_adjacency_matrices(B_true, B_ode, B_sde, save_path='results/adjacency_comparison.png'):
    """
    Plot adjacency matrices for visual comparison
    
    Args:
        B_true: True adjacency matrix
        B_ode: ODE estimated adjacency matrix
        B_sde: SDE estimated adjacency matrix
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # True graph
    im1 = axes[0].imshow(B_true, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title('True Causal Graph', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('To Variable')
    axes[0].set_ylabel('From Variable')
    plt.colorbar(im1, ax=axes[0])
    
    # ODE estimate
    im2 = axes[1].imshow(B_ode, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_title('ODE Estimate', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('To Variable')
    axes[1].set_ylabel('From Variable')
    plt.colorbar(im2, ax=axes[1])
    
    # SDE estimate
    im3 = axes[2].imshow(B_sde, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[2].set_title('SDE Estimate', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('To Variable')
    axes[2].set_ylabel('From Variable')
    plt.colorbar(im3, ax=axes[2])
    
    plt.suptitle('Causal Graph Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Adjacency matrix comparison plot saved to: {save_path}")
    plt.close()

def plot_noise_analysis(z, g_values, save_path='results/noise_analysis.png'):
    """
    Plot learned diffusion (noise) characteristics from SDE
    
    Args:
        z: Latent variables
        g_values: Diffusion values g(Z,t) at different time points
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Noise magnitude over time
    time_steps = np.arange(len(g_values))
    axes[0, 0].plot(time_steps, g_values[:, 0], linewidth=1.5, label='Dimension 1')
    if g_values.shape[1] > 1:
        axes[0, 0].plot(time_steps, g_values[:, 1], linewidth=1.5, label='Dimension 2')
    if g_values.shape[1] > 2:
        axes[0, 0].plot(time_steps, g_values[:, 2], linewidth=1.5, label='Dimension 3')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Diffusion Magnitude')
    axes[0, 0].set_title('Learned Noise Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Noise vs latent state
    axes[0, 1].scatter(z[:, 0], g_values[:, 0], alpha=0.5, s=10)
    axes[0, 1].set_xlabel('Latent Variable 1')
    axes[0, 1].set_ylabel('Diffusion Magnitude')
    axes[0, 1].set_title('State-Dependent Noise')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram of noise magnitudes
    axes[1, 0].hist(g_values.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Diffusion Magnitude')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Learned Noise')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Noise correlation matrix
    if g_values.shape[1] > 1:
        corr_matrix = np.corrcoef(g_values.T)
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1, 1].set_title('Noise Correlation Matrix')
        axes[1, 1].set_xlabel('Dimension')
        axes[1, 1].set_ylabel('Dimension')
        plt.colorbar(im, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'Single dimension', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Noise Correlation Matrix')
    
    plt.suptitle('SDE Diffusion Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Noise analysis plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    print("Visualization utilities for ODE vs SDE comparison")
    print("Import these functions in your analysis scripts")
    print("\nAvailable functions:")
    print("  - plot_trajectory_comparison()")
    print("  - plot_performance_comparison()")
    print("  - plot_adjacency_matrices()")
    print("  - plot_noise_analysis()")
