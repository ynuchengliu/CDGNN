import torch.nn as nn
from torch.nn import BatchNorm2d,  Conv2d,  Parameter
from P8.model.utils import ST_BLOCK_1
import torch
torch.set_default_dtype(torch.float32)
""" Original Edition for Adaptive Dynamic GCN """
class ActivateGraphSahe(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, recent, K, Kt):
        super(ActivateGraphSahe, self).__init__()
        tem_size = recent
        self.nodes = num_nodes
        self.block1 = ST_BLOCK_1(c_in, c_out, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK_1(c_out, c_out, num_nodes, tem_size, K, Kt)
        self.bn = BatchNorm2d(c_in, affine=False)
        self.conv1 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv2 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv3 = Conv2d(c_out, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.conv4 = Conv2d(c_out, 1, kernel_size=(1, 2), padding=(0, 0), stride=(1, 2), bias=True)
        self.h = Parameter(torch.zeros(num_nodes, dtype=torch.float32), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=1e-14)

    def forward(self, x_r, supports):
        x_r = torch.unsqueeze(x_r[:, 0, :, :], 1)
        x_r = self.bn(x_r)
        x = torch.cat((x_r,), -1)
        D_h = torch.diag_embed(self.h)
        A1 = supports + D_h
        x, _, _ = self.block1(x, A1)
        x, d_adj, t_adj = self.block2(x, A1)
        x1 = x[:, :, :, 0:12]
        x2 = x[:, :, :, 12:24]
        x1 = self.conv1(x1).squeeze()
        x2 = self.conv2(x2).squeeze()
        x = x1 + x2
        return x, d_adj, A1

