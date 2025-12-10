import torch
from torch import nn
import torch.nn.functional as F
from utils.locally_connected import LocallyConnected


class Decoder(nn.Module):
    def __init__(self, dims, z_dims, bias=False, device=None):
        super(Decoder, self).__init__()
        self.dims = dims
        self.z_dims = z_dims
        self.GL_reg = 1

        self.fc1 = nn.Linear(self.z_dims[0], self.dims[0] * self.dims[1]).to(device=device)
        layer = []
        for l in range(len(dims) - 2):
            layer.append(LocallyConnected(self.dims[0], dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layer).to(device=device)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.elu = nn.ELU(inplace=True)

        nn.init.constant_(self.fc1.weight, 0.01)

    def forward(self, z):  #  z [m, dz]
        z = self.fc1(z)  #  z [m, dx * m1]
        z = z.view(-1, self.dims[0], self.dims[1])  # z [m, dx, m1]
        for fc in self.fc2:
            z = self.elu(z)
            z = fc(z)
        x = z.squeeze()
        return x

    def m_reg(self, we):
        fc_weight = torch.abs(self.fc1.weight).view(self.dims[0], -1, self.z_dims[0])  # [d, m1, d_z]
        en_weight = torch.abs(we).view(self.z_dims[0], self.dims[0], -1)  # [d_z, d, m1]
        fc_weight = fc_weight.permute(1, 0, 2)  # [m1, d, d_z]
        en_weight = en_weight.permute(2, 0, 1)  # [m1, d_1, d]
        W = torch.matmul(fc_weight, en_weight)  # [m1, d, d]
        reg = torch.sum(W ** 2)
        return reg

    def l2_reg(self):
        reg = 0.
        # reg += torch.sum(self.fc1.weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def l1_reg(self):
        return torch.sum(torch.abs(self.fc1.weight))

    def group_weight(self, we, gamma=0.5):
        fc_weight = self.fc1.weight.view(self.dims[0], -1, self.z_dims[0])  # [d, m1, d_z]
        en_weight = we.view(self.z_dims[0], self.dims[0], -1)  # [d_z, d, m1]
        fc_weight = torch.sum(fc_weight ** 2, dim=1).pow(0.5)  # [d, d_z]
        en_weight = torch.sum(en_weight ** 2, dim=2).pow(0.5)  # [d_z, d]
        W = (fc_weight * en_weight.t()).pow(gamma)
        self.GL_reg = 1 / W  # [d, d_z]

    def proximal(self, lam=0.01, eta=0.1):
        w = self.fc1.weight  # [d, d_z*m1]
        wj = w.view(self.dims[0], -1, self.z_dims[0])  # [d, m1, d_z]
        tmp = torch.sum(wj ** 2, dim=1).pow(0.5) - lam * self.GL_reg * eta  # [d, d_z]
        alpha = torch.clamp(tmp, min=0)  # [d, d_z]
        new_w_step = F.normalize(wj, dim=1) * alpha[:, None, :]  # [d, m1, d_z]
        w.data = new_w_step.view(-1, self.z_dims[0])  # [d*m1, d_z]

    def get_adj(self):
        # get step adj
        fc_weight = self.fc1.weight  # [d * m1, d_z]
        fc_weight = fc_weight.view(self.dims[0], -1, self.z_dims[0])  # [d, m1, d_z]
        A = torch.sum(fc_weight * fc_weight, dim=1).t()  # [d_z, d]
        W = torch.sqrt(A)  # [d_z, d]
        W = W.cpu().detach().numpy()
        return W