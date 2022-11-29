import torch.nn as nn
import torch
import numpy as np



def get_ture_A_and_last_A(all_A):
    ture_A = []
    last_A = []
    for i,j in zip(all_A[3::2],all_A[2::2]):
        ture_A.append(i)
        last_A.append(j)
    ture_A = np.array(ture_A)
    last_A = np.array(last_A)
    return ture_A,last_A
class predict_A(nn.Module):
    def __init__(self,num_nodes):
        super(predict_A, self).__init__()
        self.seta = nn.Parameter(torch.FloatTensor(num_nodes,num_nodes),requires_grad=True)
        torch.nn.init.xavier_uniform_(self.seta)
    def forward(self,A_last,event_list,num_nodes):
        predict_A = []
        for i in range(len(A_last)):
            fake_A = self.seta*event_list[i]+A_last[i]
            predict_A.append(fake_A)
        temp = predict_A[0]
        for i in predict_A[1:]:
            temp = torch.cat((temp,i),0)
        temp = temp.view(len(A_last),num_nodes,num_nodes)
        return temp




