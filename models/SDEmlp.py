import torch
from torch import nn
import torch.nn.functional as F
from utils.locally_connected import LocallyConnected


class SDEmlp(nn.Module):
    """
    Neural SDE module that models both drift (deterministic) and diffusion (stochastic) terms.
    
    SDE form: dZ = f(Z,t)dt + g(Z,t)dW
    where:
    - f(Z,t) is the drift function (deterministic dynamics)
    - g(Z,t) is the diffusion function (stochastic noise term)
    - dW is Brownian motion
    """
    def __init__(self, z_dims, bias=True, device=None, noise_type='diagonal', sde_type='ito'):
        super(SDEmlp, self).__init__()
        self.z_d = z_dims[0]
        self.z_dims = z_dims
        self.GL_reg = 1
        self.noise_type = noise_type  # 'diagonal', 'scalar', 'general'
        self.sde_type = sde_type  # 'ito' or 'stratonovich'
        
        # Drift network f(Z,t)
        self.fc1_drift = nn.Linear(self.z_d, self.z_d * self.z_dims[1], bias=bias, device=device)
        layers_drift = []
        for l in range(len(z_dims) - 2):
            layers_drift.append(LocallyConnected(self.z_d, self.z_dims[l + 1], self.z_dims[l + 2], bias=bias))
        self.fc2_drift = nn.ModuleList(layers_drift).to(device=device)
        
        # Diffusion network g(Z,t)
        if noise_type == 'scalar':
            # Single scalar noise for all dimensions. Use a single input feature
            # (the mean across latent dimensions) so fc1 expects 1 input.
            self.fc1_diffusion = nn.Linear(1, self.z_dims[1], bias=bias, device=device)
            layers_diffusion = []
            for l in range(len(z_dims) - 2):
                if l == len(z_dims) - 3:  # Last layer outputs 1
                    layers_diffusion.append(LocallyConnected(1, self.z_dims[l + 1], 1, bias=bias))
                else:
                    layers_diffusion.append(LocallyConnected(self.z_d, self.z_dims[l + 1], self.z_dims[l + 2], bias=bias))
            self.fc2_diffusion = nn.ModuleList(layers_diffusion).to(device=device)
        elif noise_type == 'diagonal':
            # Diagonal diffusion matrix (independent noise per dimension)
            self.fc1_diffusion = nn.Linear(self.z_d, self.z_d * self.z_dims[1], bias=bias, device=device)
            layers_diffusion = []
            for l in range(len(z_dims) - 2):
                layers_diffusion.append(LocallyConnected(self.z_d, self.z_dims[l + 1], self.z_dims[l + 2], bias=bias))
            self.fc2_diffusion = nn.ModuleList(layers_diffusion).to(device=device)
        elif noise_type == 'general':
            # Full diffusion matrix g(Z,t) in R^{d x d}
            self.fc1_diffusion = nn.Linear(self.z_d, self.z_d * self.z_d * self.z_dims[1], bias=bias, device=device)
            layers_diffusion = []
            for l in range(len(z_dims) - 2):
                layers_diffusion.append(LocallyConnected(self.z_d * self.z_d, self.z_dims[l + 1], self.z_dims[l + 2], bias=bias))
            self.fc2_diffusion = nn.ModuleList(layers_diffusion).to(device=device)
        else:
            # Fallback: create a diagonal-style diffusion to avoid attribute errors
            self.fc1_diffusion = nn.Linear(self.z_d, self.z_d * self.z_dims[1], bias=bias, device=device)
            self.fc2_diffusion = nn.ModuleList([]).to(device=device)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.elu = nn.ELU(inplace=True)
        self.softplus = nn.Softplus()  # For ensuring positive diffusion

        nn.init.constant_(self.fc1_drift.weight, 0.01)
        # initialize diffusion weights if present
        if hasattr(self, 'fc1_diffusion') and self.fc1_diffusion is not None:
            try:
                nn.init.constant_(self.fc1_diffusion.weight, 0.01)
            except Exception:
                pass

    def f(self, t, z):
        """Drift function f(Z,t)"""
        z_drift = self.fc1_drift(z)
        z_drift = z_drift.view(-1, self.z_d, self.z_dims[1])
        for fc in self.fc2_drift:
            z_drift = fc(self.elu(z_drift))
        z_drift = z_drift.squeeze(dim=2)
        z_drift = z_drift.unsqueeze(dim=1)
        return z_drift

    def g(self, t, z):
        """Diffusion function g(Z,t)"""
        if self.noise_type == 'scalar':
            # Scalar diffusion
            z_diff = torch.mean(z, dim=1, keepdim=True)  # [batch, 1]
            z_diff = self.fc1_diffusion(z_diff)
            z_diff = z_diff.view(-1, 1, self.z_dims[1])
            for fc in self.fc2_diffusion:
                z_diff = fc(self.elu(z_diff))
            z_diff = self.softplus(z_diff.squeeze(dim=2))  # Ensure positive
            # Broadcast to all dimensions
            z_diff = z_diff.unsqueeze(dim=1).expand(-1, self.z_d, -1)  # [batch, z_d, 1]
            return z_diff
        
        elif self.noise_type == 'diagonal':
            # Diagonal diffusion
            z_diff = self.fc1_diffusion(z)
            z_diff = z_diff.view(-1, self.z_d, self.z_dims[1])
            for fc in self.fc2_diffusion:
                z_diff = fc(self.elu(z_diff))
            z_diff = self.softplus(z_diff.squeeze(dim=2))  # Ensure positive
            z_diff = z_diff.unsqueeze(dim=2)  # [batch, z_d, 1]
            return z_diff
        
        elif self.noise_type == 'general':
            # General diffusion matrix
            z_diff = self.fc1_diffusion(z)
            z_diff = z_diff.view(-1, self.z_d * self.z_d, self.z_dims[1])
            for fc in self.fc2_diffusion:
                z_diff = fc(self.elu(z_diff))
            z_diff = z_diff.squeeze(dim=2)
            z_diff = z_diff.view(-1, self.z_d, self.z_d)  # [batch, z_d, z_d]
            return z_diff

    def forward(self, t, z):
        """
        For compatibility with ODE solvers during inference.
        Returns only the drift term.
        """
        return self.f(t, z)

    def m_reg(self):
        """Regularization for drift network"""
        reg = 0.0
        reg += torch.sum(self.fc1_drift.weight ** 2)
        return reg
    
    def m_reg_diffusion(self):
        """Regularization for diffusion network"""
        reg = 0.0
        if hasattr(self, 'fc1_diffusion') and self.fc1_diffusion is not None:
            try:
                reg += torch.sum(self.fc1_diffusion.weight ** 2)
            except Exception:
                reg += 0.0
        return reg

    def l2_reg(self):
        """L2 regularization for all layers"""
        reg = 0.0
        for fc in self.fc2_drift:
            reg += torch.sum(fc.weight ** 2)
        for fc in self.fc2_diffusion:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def l1_reg(self):
        """L1 regularization for drift network"""
        return torch.sum(torch.abs(self.fc1_drift.weight))

    def group_weight(self, gamma=0.5):
        """Group lasso for drift network"""
        fc1_weight = self.fc1_drift.weight.view(self.z_d, -1, self.z_d)
        weights = torch.sum(fc1_weight ** 2, dim=1).pow(gamma).data
        self.GL_reg = 1 / weights

    def proximal(self, lam=0.01, eta=0.1):
        """Proximal operator for sparsity in drift network"""
        w = self.fc1_drift.weight
        wj = w.view(self.z_d, -1, self.z_d)
        tmp = torch.sum(wj * wj, dim=1).pow(0.5) - lam * self.GL_reg * eta
        alpha = torch.clamp(tmp, min=0)
        new_w_step = F.normalize(wj, dim=1) * alpha[:, None, :]
        w.data = new_w_step.view(-1, self.z_d)

    def get_adj(self):
        """Get adjacency matrix from drift network"""
        fc1_weight = self.fc1_drift.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(self.z_d, -1, self.z_d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W
    
    def sample_noise(self, z, dt):
        """
        Sample Brownian motion increment dW ~ N(0, dt)
        
        Args:
            z: latent state [batch, z_d]
            dt: time increment (scalar or tensor)
        
        Returns:
            dW: Brownian increment [batch, z_d, 1]
        """
        batch_size = z.shape[0]
        
        # Convert dt to tensor if it's a scalar
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt, device=z.device, dtype=z.dtype)
        
        sqrt_dt = torch.sqrt(dt)
        
        if self.noise_type == 'scalar':
            dW = torch.randn(batch_size, 1, 1, device=z.device, dtype=z.dtype) * sqrt_dt
            dW = dW.expand(-1, self.z_d, -1)
        elif self.noise_type == 'diagonal':
            dW = torch.randn(batch_size, self.z_d, 1, device=z.device, dtype=z.dtype) * sqrt_dt
        elif self.noise_type == 'general':
            dW = torch.randn(batch_size, self.z_d, 1, device=z.device, dtype=z.dtype) * sqrt_dt
        return dW
