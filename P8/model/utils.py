import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from scipy.sparse import csr_matrix
from torch.nn import  Conv1d, Conv2d,LayerNorm, BatchNorm1d
torch.set_default_dtype(torch.float32)


"""
    x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv_d1 = Conv1d(tem_size, tem_size, kernel_size=(2, ), dilation=(2, ), padding=(1, ), bias=False)
        self.conv_d2 = Conv1d(c_in, c_in,         kernel_size=(2, ), dilation=(2, ), padding=(1, ), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn = BatchNorm1d(tem_size)
        A = np.zeros((24, 24))
        for i in range(12):
            for j in range(12):
                A[i, j] = 1
                A[i + 12, j + 12] = 1

        self.B = (torch.tensor((-1e13) * (1 - A))).type(torch.float32).cuda(1)

    def forward(self, seq):
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n
        f1 = self.conv_d1(f1)
        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()  # b,c,l
        f2 = self.conv_d2(f2)
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = self.bn(logits).permute(0, 2, 1).contiguous()
        coefs = torch.softmax(logits + self.B, -1)
        return coefs




class Graph_Sage(nn.Module):

    def __init__(self,num_layer):
        super(Graph_Sage, self).__init__()

        self.graphsage = nn.ModuleList()
        for i in range(num_layer):
            self.graphsage.append(SAGEConv(24, 24, 'mean').cuda(1))


    def forward(self, x, adj):
        nSample, feat_in, nNode, length = x.shape
        Ls = []
        L1 = adj
        L1 = L1.cpu()
        L1 = L1.detach().numpy()
        L0 = np.eye(nNode)*4
        Ls.append(L0)
        Ls.append(L1)
        Ls = np.array(Ls)
        Ls = torch.Tensor(Ls).cuda(1)
        adj = adj.cpu()
        adj = adj.detach().numpy()
        mat = csr_matrix(adj)
        g = dgl.from_scipy(mat)
        g = g.to(torch.device('cuda:1'))
        adj = torch.FloatTensor(adj).cuda(1)
        x = torch.einsum('bcnl,knq->bckql', x, Ls).contiguous()
        res = x.permute(3, 1, 2, 0, 4)
        for layer in self.graphsage:
            res = layer(g, res)
        res = res.permute(3, 1, 2, 0, 4)
        x = res.view(nSample, -1, nNode, length)

        return x
class ST_BLOCK_1(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_1, self).__init__()

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.TATT_1 = TATT_1(c_out, num_nodes, tem_size)
        self.graph_sage = Graph_Sage(num_layer=2)
        self.K = K
        self.time_conv = Conv2d(c_in, c_out, kernel_size=(1, Kt), padding=(0, 1), stride=(1, 1), bias=True)
        self.c_out = c_out
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        x_1 = self.time_conv(x)
        x_1 = self.bn(x_1)
        x_1 = F.leaky_relu(x_1)
        x_1 = self.graph_sage(x_1, supports)
        filter, gate = torch.split(x_1, [self.c_out, self.c_out], 1)
        x_1 = torch.sigmoid(gate) * F.leaky_relu(filter)
        T_coef = self.TATT_1(x_1)
        T_coef = T_coef.transpose(-1, -2)
        x_1 = torch.einsum('bcnl,blq->bcnq', x_1, T_coef)
        out = self.bn(F.leaky_relu(x_1) + x_input)
        return out, supports, T_coef




