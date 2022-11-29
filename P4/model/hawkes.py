
import numpy as  np
import torch
from torch import nn







def get_event_list(all_A,event_flag):
    event_list = []
    for i,j in zip(all_A[:-1],all_A[1:]):
        event = j - i
        event[event>=event_flag] = 1
        event[event<=-event_flag] = -1
        event[(event<event_flag)&(event>-event_flag)] =0
        event_list.append(event)
    event_list = np.array(event_list)
    return event_list

def get_event_train_and_event_ture(event_list,batch_size):
    event_ture = []
    for j in event_list[batch_size::batch_size]:
        event_ture.append(j)
    event_ture = np.array(event_ture)
    return event_list,event_ture

def get_u_begin(train_A):
    u=[]
    u_list = []
    num_nodes = len(train_A[0])
    temp=len(train_A[train_A==0])/(len(train_A)*num_nodes*num_nodes)
    temp2 = (1-temp)/2
    u.append(temp)
    u.append(temp2)
    u.append(temp2)
    u = np.array(u)
    for i in range(num_nodes*num_nodes):
        u_list.append(u)
    u_list = np.array(u_list)
    u_list=u_list.reshape(num_nodes,num_nodes,3)
    return u_list
class hawkes_parameter(nn.Module):
    def __init__(self,num_nodes,num_event_type):
        super(hawkes_parameter, self).__init__()
        self.input_w=nn.Parameter(torch.FloatTensor(num_nodes,num_nodes,num_event_type),requires_grad=True)
        self.input_u = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        self.input_d=nn.Parameter(torch.FloatTensor(num_nodes,num_nodes,num_event_type),requires_grad=True)
        self.forget_w=nn.Parameter(torch.FloatTensor(num_nodes,num_nodes,num_event_type),requires_grad=True)
        self.forget_u=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        self.forget_d=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes,num_event_type), requires_grad=True)
        self.output_w=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes,num_event_type), requires_grad=True)
        self.output_u=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        self.output_d=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes,num_event_type), requires_grad=True)
        self.limit_w = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.limit_u = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.limit_d = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.input_w2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.input_u2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        self.input_d2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.forget_w2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.forget_u2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        self.forget_d2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.output_w2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.output_u2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        self.output_d2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.limit_w2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.limit_u2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.limit_d2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()
        nn.init.uniform_(self.input_w, a=0, b=1e-4)
        nn.init.uniform_(self.input_u, a=0, b=1e-4)
        nn.init.uniform_(self.input_d, a=0, b=1e-4)
        nn.init.uniform_(self.output_u, a=0, b=1e-4)
        nn.init.uniform_(self.output_w, a=0, b=1e-4)
        nn.init.uniform_(self.output_d, a=0, b=1e-4)
        nn.init.uniform_(self.forget_u, a=0, b=1e-4)
        nn.init.uniform_(self.forget_d, a=0, b=1e-4)
        nn.init.uniform_(self.forget_w, a=0, b=1e-4)
        nn.init.uniform_(self.limit_w, a=0, b=1e-4)
        nn.init.uniform_(self.limit_u, a=0, b=1e-4)
        nn.init.uniform_(self.limit_d, a=0, b=1e-4)
        nn.init.uniform_(self.input_u2, a=0, b=1e-4)
        nn.init.uniform_(self.input_d2, a=0, b=1e-4)
        nn.init.uniform_(self.input_w2, a=0, b=1e-4)
        nn.init.uniform_(self.output_u2, a=0, b=1e-4)
        nn.init.uniform_(self.output_d2, a=0, b=1e-4)
        nn.init.uniform_(self.output_w2, a=0, b=1e-4)
        nn.init.uniform_(self.forget_u2, a=0, b=1e-4)
        nn.init.uniform_(self.forget_d2, a=0, b=1e-4)
        nn.init.uniform_(self.forget_w2, a=0, b=1e-4)
        nn.init.uniform_(self.limit_d2, a=0, b=1e-4)
        nn.init.uniform_(self.limit_u2, a=0, b=1e-4)
        nn.init.uniform_(self.limit_w2, a=0, b=1e-4)



    def forward(self,history_event1,u_begin1,history_event2,u_begin2):
        ht1 = self.tanh(u_begin1)
        in_put = self.sigmoid(torch.matmul(history_event1, self.input_w) + torch.matmul(self.input_u, ht1) + self.input_d)
        out_put = self.sigmoid(torch.matmul(history_event1,self.output_w ) + torch.matmul(self.output_u ,ht1) + self.output_d)
        for_get = self.sigmoid(torch.matmul(history_event1,self.forget_w)+ torch.matmul(self.forget_u ,ht1)+ self.forget_d)
        z = self.tanh(torch.matmul(history_event1, self.limit_w ) + self.limit_u * ht1 + self.limit_d)
        ht2 = self.tanh(u_begin2)
        in_put2 = self.sigmoid(torch.matmul(history_event2, self.input_w2) + torch.matmul(self.input_u2, ht2) + self.input_d2)
        out_put2 = self.sigmoid(torch.matmul(history_event2, self.output_w2) + torch.matmul(self.output_u2, ht2) + self.output_d2)
        for_get2 = self.sigmoid(torch.matmul(history_event2, self.forget_w2) + torch.matmul(self.forget_u2, ht2) + self.forget_d2)
        z2 = self.tanh(torch.matmul(history_event2, self.limit_w2) + self.limit_u2 * ht2 + self.limit_d2)
        ct = for_get * ht1+ in_put * z
        ct2 = for_get2 * ht2+ in_put2 * z2
        return ct,out_put,ct2,out_put2

class generator(nn.Module):
    def __init__(self,num_nodes,event_list_length):
        super(generator, self).__init__()
        # self.a =nn.Parameter(torch.FloatTensor(2,num_nodes,num_nodes,3),requires_grad=True)
        self.w1 = nn.Parameter(torch.FloatTensor(num_nodes, 1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(num_nodes, 1), requires_grad=True)
        self.w = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        self.liner=nn.Linear(3,1)
        self.hakwkes_paramet=hawkes_parameter(num_nodes,3)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softplus=torch.nn.Softplus()
        nn.init.uniform_(self.w2, a=0, b=1e-2)
        nn.init.uniform_(self.w, a=0, b=1e-2)
        nn.init.uniform_(self.w1, a=0, b=1e-2)
    def forward(self, time,history_event,u_begin1,u_begin2,batch_size,num_nodes):
        ct1,ot1,ct2, ot2=self.hakwkes_paramet(history_event[0],u_begin1,history_event[1], u_begin2)
        q1=-torch.matmul(history_event[0],self.w1)
        q2=-torch.matmul(history_event[1],self.w2)
        cell=ct2+(ct1-ct2)*(torch.exp((q1)*time[0])+torch.exp(q2*time[1]))
        u_begin1=ot1*self.tanh(cell)
        u_begin2=ot2*self.tanh(cell)
        lamda=self.softplus(torch.matmul(self.w,cell))
        lamda2=self.liner(lamda).squeeze()
        return lamda2,u_begin1,u_begin2
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.lose_functional = torch.nn.MSELoss()
    def forward(self, ture_event ,fake_event):
        loss = self.lose_functional(fake_event,ture_event)
        loss.backward()
        return loss

