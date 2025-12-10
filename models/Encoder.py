import torch
from torch import nn
import torch.nn.functional as F
from utils.locally_connected import LocallyConnected


class Encoder(nn.Module):
    def __init__(self, dims, z_dims, bias=False, device=None):
        super(Encoder, self).__init__()
        self.dims = dims
        self.z_dims = z_dims
        self.GL_reg = 1
        layer = []
        for l in range(len(dims) - 2):
           layer.append(LocallyConnected(self.dims[0], dims[-1 - l], dims[-2 - l], bias=bias))
        self.fc1 = nn.ModuleList(layer).to(device=device)
        self.fc_mu = nn.Linear((self.dims[0]) * self.dims[1], self.z_dims[0], bias=bias, device=device)
        self.fc_logvar = nn.Linear((self.dims[0]) * self.dims[1], self.z_dims[0], bias=bias, device=device)
        self.leaky_relu = nn.LeakyReLU(negative_slope=5)
        self.elu = nn.ELU(inplace=True)

        nn.init.constant_(self.fc_mu.weight, 0.01)
        nn.init.zeros_(self.fc_logvar.weight)

    def forward(self, x):
        x = x.view(-1, self.dims[0], self.dims[-1]) # [n, d, 1]
        for fc in self.fc1:
            x = fc(x)
            x = self.elu(x)  # [n, d, m1]
        x = x.view(-1, (self.dims[0]) * self.dims[1])  # [n, d*m1]

        mu = self.fc_mu(x)  # [n, d_z]
        logvar = self.fc_logvar(x)  # [n, d_z]

        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z

    def l2_reg(self):
        reg = 0.
        for fc in self.fc1:
            reg += torch.sum(fc.weight ** 2)
        # reg += torch.sum(self.fc_mu.weight ** 2)
        reg += torch.sum(self.fc_logvar.weight ** 2)
        return reg

    def l1_reg(self):
        return torch.sum(torch.abs(self.fc_mu.weight))

    def m_reg(self, wd):
        fc2_weight = torch.abs(self.fc_mu.weight).view(self.z_dims[0], self.dims[0], -1)  # [d_z, d, m1]
        de_weight = torch.abs(wd).view(self.dims[0], -1, self.z_dims[0])  # [d, m1, d_z]
        fc2_weight = fc2_weight.permute(2, 0, 1)  # [m1, d_z, d]
        de_weight = de_weight.permute(1, 0, 2)  # [m1, d, d_z]
        W = torch.matmul(fc2_weight, de_weight)  # [m1, d_z, d_z]
        reg = torch.sum(W ** 2)
        return reg

    def group_weight(self, wd, gamma=0.5):
        fc2_weight = self.fc_mu.weight.view(self.z_dims[0], self.dims[0], -1)  # [d_z, d, m1]
        de_weight = wd.view(self.dims[0], -1, self.z_dims[0])  # [d, m1, d_z]
        fc2_weight = torch.sum(fc2_weight ** 2, dim=2).pow(0.5)  # [d_z, d]
        de_weight = torch.sum(de_weight ** 2, dim=1).pow(0.5)  # [d, d_z]
        W = (fc2_weight * de_weight.t()).pow(gamma)
        self.GL_reg = 1 / W  # [d_z, d]

    def proximal(self, lam=0.01, eta=0.1):
        w = self.fc_mu.weight  # [d_z, d*m1]
        wj = w.view(self.z_dims[0], self.dims[0], -1)  # [d_z, d, m1]
        tmp = torch.sum(wj ** 2, dim=2).pow(0.5) - lam * self.GL_reg * eta  # [d_z, d]
        alpha = torch.clamp(tmp, min=0)  # [d_z, d]
        new_w_step = F.normalize(wj, dim=2) * alpha[:, :, None]  # [d_z, d, m1]
        w.data = new_w_step.view(self.z_dims[0], -1)  # [d_z, d*m1]

    def get_adj(self):
        # get step adj
        fc_weight = self.fc_mu.weight  # [d_z, d * m1]
        fc_weight = fc_weight.view(self.z_dims[0], self.dims[0], -1)  # [d_z, d, m1]
        A = torch.sum(fc_weight * fc_weight, dim=2).t()  # [d, d_z]
        W = torch.sqrt(A)  # [d, d_z]
        W = W.cpu().detach().numpy()
        return W
