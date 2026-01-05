import logging
import os
import random

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from torchmetrics import SpearmanCorrCoef

from models.ControlPolicy import ControlPolicy
from models.Decoder import Decoder
from models.Encoder import Encoder
from models.SDEmlp import SDEmlp

from utils.evaluation import compute_auroc


class CausalSDE:
    def __init__(self, args):
        self.dims = args.dims
        self.z_dims = args.z_dims
        self.delta_t = args.delta_t
        self.obs_delta_t = getattr(args, 'obs_delta_t', args.delta_t)
        self.annealing_proximal = getattr(args, 'annealing_proximal', False)
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.lambda4 = args.lambda4
        self.lambda5 = getattr(args, 'lambda5', 0.01)  # Regularization for diffusion
        self.gamma = args.gamma
        self.bias = args.bias
        self.irregular = args.irregular
        self.lr = args.lr
        self.train_steps = args.train_steps
        self.pretrain_steps = args.pretrain_steps
        self.lasso_type = args.lasso_type
        self.w_threshold = args.w_threshold
        self.device_type = args.device_type
        self.device_ids = args.device_ids
        self.noise_type = getattr(args, 'noise_type', 'diagonal')  # 'scalar', 'diagonal', 'general'
        self.sde_type = getattr(args, 'sde_type', 'ito')  # 'ito' or 'stratonovich'
        self.seed = getattr(args, 'seed', None)
        
        self.beta_control = args.beta_control
        self.control_context = args.control_context
        self.control_hidden = args.control_hidden
        self.control_layers = args.control_layers
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda:'+str(self.device_ids))
        else:
            device = torch.device('cpu')
        self.device = device
        self.encoder = Encoder(self.dims, self.z_dims, self.bias, self.device)
        self.sdemlp = SDEmlp(self.z_dims, self.bias, self.device, 
                            noise_type=self.noise_type, sde_type=self.sde_type)
        self.decoder = Decoder(self.dims, self.z_dims, self.bias, self.device)

        if self.control_context == "none":
            context_dim = 0
        elif self.control_context == "observation":
            context_dim = self.dims[0]
        elif self.control_context == "latent":
            context_dim = self.z_dims[0]

        self.control_policy = ControlPolicy(
            z_dim=self.z_dims[0],
            hidden_dim=self.control_hidden,
            context_dim=context_dim,
            num_hidden_layers=self.control_layers,
            bias=self.bias,
            device=self.device,
        )

    def squared_loss(self, output, target):
        n = target.shape[0]
        loss = 0.5 / n * torch.sum((output - target) ** 2)
        return loss

    def kl_loss(self, mu1, mu2, logvar):
        n = logvar.shape[0]
        kl = 0.5 / n * (torch.sum((mu1 - mu2) ** 2) + torch.sum(torch.exp(logvar) - 1 - logvar))
        return kl

    def _apply_control(self, diffusion: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        if diffusion.dim() == 3:
            print(diffusion.shape, "diffusion")
            print(control.shape, "control")
            return (diffusion.squeeze(-1) * control)
        return diffusion * control
    
    def _control_energy(self, control: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        effort = torch.sum(control * control, dim=1)
        return 0.5 * torch.mean(effort * dt)

    def euler_maruyama_step(self, z, context, t, dt):
        """
        Euler-Maruyama method for SDE integration
        dZ = f(Z,t)dt + g(Z,t)dW
        Z_{t+dt} = Z_t + f(Z_t,t)*dt + g(Z_t,t)*sqrt(dt)*N(0,1)
        """
        drift = self.sdemlp.f(t, z)  # [batch, 1, z_d] or [batch, z_d, 1]
        diffusion = self.sdemlp.g(t, z)  # [batch, z_d, 1] or [batch, z_d, z_d]
        dW = self.sdemlp.sample_noise(z, dt)  # [batch, z_d, 1]

        control = self.control_policy(t, z, context=context)
        # control = self._apply_control(diffusion, control)
        
        # Compute stochastic increment
        if self.noise_type == 'general':
            # For general diffusion: g(Z,t) @ dW
            stochastic_term = torch.bmm(diffusion, dW).squeeze(dim=2)  # [batch, z_d]
        else:
            # For diagonal or scalar: element-wise multiplication
            stochastic_term = (diffusion * dW).squeeze(dim=2)  # [batch, z_d]
        
        # Update: Z_{t+dt} = Z_t + drift*dt + stochastic_term
        z_next = z + (drift.squeeze(dim=1) + control) * dt + stochastic_term

        control_energy = self._control_energy(control, dt)
        
        return z_next, control_energy

    def learn(self, X, times, Z_orig, B):
        X_torch = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        T_torch = torch.as_tensor(times, dtype=torch.float32).to(self.device)

        # ============= Phase 1: Pretrain VAE (Encoder + Decoder) =============
        vae_optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.decoder.parameters()}
        ], lr=self.lr)
        
        print(f"Pretraining VAE for {self.pretrain_steps} steps...")
        pretrain_t = tqdm(range(self.pretrain_steps), desc="VAE Pretrain")
        for k in pretrain_t:
            vae_optimizer.zero_grad()
            mu, logvar, Z = self.encoder(X_torch)
            X_hat = self.decoder(Z)
            
            # VAE loss: reconstruction + KL divergence (standard VAE prior)
            recon_loss = self.squared_loss(X_hat, X_torch)
            # KL divergence with standard normal prior: KL(q(z|x) || N(0,I))
            kl_prior = 0.5 / X_torch.shape[0] * torch.sum(
                mu.pow(2) + torch.exp(logvar) - 1 - logvar
            )
            vae_loss = recon_loss + kl_prior
            
            pretrain_t.set_postfix(loss=vae_loss.item(), recon=recon_loss.item(), kl=kl_prior.item())
            vae_loss.backward()
            vae_optimizer.step()
        
        print("VAE pretraining complete. Starting main training...")
        
        # ============= Phase 2: Main Training (all components) =============
        optimizer = torch.optim.Adam([{'params': self.encoder.parameters()},
                                           {'params': self.decoder.parameters()},
                                          {'params': self.sdemlp.parameters()},
                                          {'params': self.control_policy.parameters()}], lr=self.lr)
        optimizer = optimizer

        t = tqdm(range(self.train_steps))
        for k in t:
            # LINEAR ANNEALING: Increase Proximal penalty after step 1000
            if self.annealing_proximal == True and k > 1000:
                # Slowly ramp from 0.01 to 0.04
                progress = (k - 1000) / (self.train_steps - 1000)
                self.lambda3 = 0.01 + (0.03 * progress)
                self.lambda4 = 0.01 + (0.03 * progress)

            # Variable learning rate: decrease by 0.002 every 8000 steps
            # (subtractive schedule). Ensure lr doesn't go below 0.
            steps_per_drop = 3000
            drop_amount = 0.001
            num_drops = k // steps_per_drop
            new_lr = max(self.lr - num_drops * drop_amount, 0.001)
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            optimizer.zero_grad()
            mu, logvar, Z = self.encoder(X_torch)
            X_hat = self.decoder(Z)
            
            if self.irregular == 'irregular':
                # 1. Calculate time gaps between consecutive observations
                # T_torch shape is [N], gaps shape is [N-1]
                gaps = T_torch[1:] - T_torch[:-1]
                
                max_gap = int(torch.max(gaps).item())

                mu1 = []
                mu2 = []
                control_energy = 0.0
                
                # Pass appropriate context based on control_context setting
                if self.control_context == 'observation':
                    context = X_torch
                elif self.control_context == 'latent':
                    context = Z
                else:
                    context = None
                
                # Calculate how many small steps make up one integer observation unit
                # e.g., 1.0 / 0.005 = 200 steps
                steps_per_obs_gap = int(round(self.obs_delta_t / self.delta_t))

                # 2. Iterate through strictly necessary step sizes (1, 2, ... max_gap)
                for i in range(1, max_gap + 1):
                    # Find indices 'k' where the gap between k and k+1 is exactly 'i' steps
                    # This returns indices valid for mu[:-1]
                    idx_transitions = torch.where(gaps == i)[0]
                    
                    if len(idx_transitions) == 0:
                        continue

                    # 3. Select Start (Z_t) and Target (Z_{t+i})
                    # idx_transitions are from 0 to N-2, so +1 is always safe (<= N-1)
                    index_mu = mu[idx_transitions]      # Z_t
                    index_mu1 = mu[idx_transitions + 1] # Z_{t+next} (Target)
                    
                    current_context = context[idx_transitions] if context is not None else None

                    total_steps = i * steps_per_obs_gap

                    # 4. Integrate SDE over 'i' steps
                    for j in range(total_steps):
                        index_mu, step_energy = self.euler_maruyama_step(index_mu, context=current_context, t=j, dt=self.delta_t)
                        control_energy += step_energy.sum()
                    
                    mu1.append(index_mu1)
                    mu2.append(index_mu)
                
                if len(mu1) > 0:
                    mu1 = torch.vstack(mu1)
                    mu2 = torch.vstack(mu2)
                    kl_loss = self.kl_loss(mu1, mu2, logvar)
                else:
                    # Fallback if batch has no valid transitions (e.g. size 1)
                    kl_loss = torch.tensor(0.0, device=self.device)
                    control_energy = torch.tensor(0.0, device=self.device)
                
            elif self.irregular == 'sparse':
                # gaps hold all the time gaps between observations
                gaps = T_torch[1:] - T_torch[:-1]
                max_gap = torch.max(gaps).int()
                
                mu2 = mu
                control_energy = 0.0
                
                # Pass appropriate context based on control_context setting
                if self.control_context == 'observation':
                    context = X_torch
                elif self.control_context == 'latent':
                    context = Z
                else:
                    context = None

                # Integrate SDE over multiple smaller time steps instead of one large step
                for i in range(max_gap):
                    mu2, step_energy = self.euler_maruyama_step(mu2, context=context, t=i, dt=self.delta_t)
                    control_energy += step_energy
                kl_loss = self.kl_loss(mu[1:], mu2[:-1], logvar)
                
            else:
                # Frequent/regular sampling
                # Pass appropriate context based on control_context setting
                if self.control_context == 'observation':
                    context = X_torch
                elif self.control_context == 'latent':
                    context = Z
                else:
                    context = None
                mu2, control_energy = self.euler_maruyama_step(mu, context=context, t=T_torch, dt=self.delta_t)
                kl_loss = self.kl_loss(mu[1:], mu2[:-1], logvar)
            
            # Reconstruction loss
            loss = self.squared_loss(X_hat, X_torch)
            
            # Regularization terms
            m_reg = 0.5 * self.lambda1 * (self.encoder.m_reg(self.decoder.fc1.weight) + 
                                          self.decoder.m_reg(self.encoder.fc_mu.weight))
            l2_reg_drift = 0.5 * self.lambda2 * self.sdemlp.m_reg()
            l2_reg_diffusion = 0.5 * self.lambda5 * self.sdemlp.m_reg_diffusion()
            
            # Control energy term
            control_energy = self.beta_control * control_energy

            obj = loss + kl_loss + m_reg + l2_reg_drift + l2_reg_diffusion + control_energy
            t.set_postfix(loss=obj.item(), recon=loss.item(), kl=kl_loss.item(), 
                            control=control_energy.item(), lr=new_lr)
            obj.backward()
            optimizer.step()

            if self.lasso_type == 'AGL':
                self.encoder.group_weight(self.decoder.fc1.weight, gamma=self.gamma)
                self.sdemlp.group_weight(gamma=self.gamma)
                self.decoder.group_weight(self.encoder.fc_mu.weight, gamma=self.gamma)
            
            self.encoder.proximal(lam=self.lambda3, eta=0.01)
            self.sdemlp.proximal(lam=self.lambda4, eta=0.01)
            self.decoder.proximal(lam=self.lambda3, eta=0.01)

            if (k + 1) % 100 == 0:
                # Evaluate latent recovery
                W_corrs, B_corrs = self.get_MCC(X, Z_orig)

                # Get causal adjacency estimates
                GC_est, W_xz, W_z, W_zx = self.get_adj()

                # Evaluate causal discovery in latent space
                B_est = np.matmul(np.matmul(B_corrs, W_z), B_corrs.T)
                
                # AUROC calculation
                auroc = compute_auroc(B, B_est, diagonal=True)
                print(f"\nStep {k+1}: Latent MCC: {W_corrs:.4f}, AUROC: {auroc:.4f}")

    def get_MCC(self, X, Z, method='pearson'):
        X_torch = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        Z_torch = torch.as_tensor(Z, dtype=torch.float32).to(self.device)
        mu, logvar, est_Z = self.encoder(X_torch)
        if method=='pearson':
            W_corrs = torch.zeros((Z_torch.shape[1], est_Z.shape[1])).to(self.device)
            for i in range(Z_torch.shape[1]):
                for j in range(est_Z.shape[1]):
                    data = torch.stack([Z_torch[:, i], est_Z[:, j]], dim=1).T
                    W_corrs[i, j] = torch.abs(torch.corrcoef(data)[0, 1])
        elif method=='spearman':
            spearman = SpearmanCorrCoef(num_outputs=Z_torch.shape[1])
            W_corrs = torch.abs(spearman(est_Z, Z_torch))
        max_index = linear_sum_assignment(-1 * W_corrs.cpu().detach().numpy())
        MCC = np.sum(W_corrs[max_index].cpu().detach().numpy()) / self.z_dims[0]
        B_corrs = np.zeros((self.z_dims[0], self.z_dims[0]))
        B_corrs[max_index] = 1
        return MCC, B_corrs

    def get_adj(self):
        W_xz = self.encoder.get_adj()
        B_xz = (((W_xz - np.min(W_xz, axis=0)) / (np.max(W_xz, axis=0) - np.min(W_xz, axis=0) + 1e-5)) > 0.1).astype(int)
        
        W_z = self.sdemlp.get_adj()
        B_z = (((W_z - np.min(W_z, axis=0)) / (np.max(W_z, axis=0) - np.min(W_z, axis=0) + 1e-5)) > self.w_threshold).astype(int)
        
        W_zx = self.decoder.get_adj()
        B_zx = (((W_zx - np.min(W_zx, axis=1, keepdims=True)) / (np.max(W_zx, axis=1, keepdims=True) - np.min(W_zx, axis=1, keepdims=True) + 1e-5)) > 0.1).astype(int)
        
        B_est = B_xz @ B_z @ B_zx
        B_est = (abs(B_est) > 0).astype(int)
        return B_est, B_xz, B_z, B_zx
