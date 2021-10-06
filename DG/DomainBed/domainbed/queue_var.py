import torch
import torch.nn as nn
from torch.nn import functional as F

import copy
import math

##### Variables to set for RCERM #####
queue_sz=100 # the memory module/ queue size
tau = 0.05 # temperature parameter in the objective
momentum = 0.999 # theta in momentum encoding step
train_queues=[]
##### Variables to set for RCERM #####

def not_ind_i(lst,ind_i): # returns all elements in list except pointed by the index ind_i
    if ind_i==0:
        return lst[ind_i+1:]
    elif ind_i==len(lst)-1:
        return lst[:-1]
    else:
        res=lst[:ind_i]
        res.extend(lst[ind_i+1:])
        return res
    
def get_pos_neg_queues(id_c,id_d,train_queues):
    ind_class_other_domains=not_ind_i(train_queues[id_c],id_d) # indexed class, other domains; len = N_d-1
    positive_queue=None
    for positive_domain_queue in ind_class_other_domains:
        if positive_queue is None:
            positive_queue=positive_domain_queue
        else:
            positive_queue = torch.cat((positive_queue, positive_domain_queue), 0)
    #print('Positive Queue Generated for class ',id_c,' domain ',id_d,' with size ',positive_queue.size())

    other_classes=not_ind_i(train_queues,id_c) # remaining classes; len = N_c-1
    negative_queue=None

    for negative_class in other_classes:
        for negative_domain_queue in negative_class:
            if negative_queue is None:
                negative_queue=negative_domain_queue
            else:
                negative_queue = torch.cat((negative_queue, negative_domain_queue), 0)
    #print('Negative Queue Generated for class ',id_c,' domain ',id_d,' with size ',negative_queue.size())
    return positive_queue,negative_queue

def loss_function(q, k, queue):

    N = q.shape[0]
    C = q.shape[1]

    pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),tau))
    neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),tau)), dim=1)
    denominator = neg + pos # if we omit +pos, loss would reduce to -ve values.

    return torch.mean(-torch.log(torch.div(pos,denominator))) # this loss is essentially a reformulation of infoNCE loss

class AttenHead(nn.Module):
    def __init__(self, fdim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.fatt = fdim//num_heads

        for i in range(num_heads):
            setattr(self, f'embd{i}', nn.Linear(fdim, self.fatt))
        for i in range(num_heads):
            setattr(self, f'fc{i}', nn.Linear(2*self.fatt, self.fatt))
        self.fc = nn.Linear(self.fatt*num_heads, fdim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fx_in, fp_in):
        fp_in = fp_in.squeeze(0)# Return tensor with all  dimensions of input of size 1 removed. 
        d = math.sqrt(self.fatt)

        Nx = len(fx_in)
        
#         print(fx_in.size(),fp_in.size())
        f = torch.cat([fx_in, fp_in])
        f = torch.stack([getattr(self, f'embd{i}')(f) for i in range(self.num_heads)])  # head x N x fatt
        # f: torch.Size([1, 5, 6]), ie., [Nheads,(Nfx+Nfp),#dims] 
        # fx: torch.Size([1, 6]), i.e, [Nfx,#dims] 
        # fp: torch.Size([1, 4, 6]), i.e., [Nheads,Nfp,#dims] 
        fx, fp = f[:, :Nx], f[:, Nx:]

        w = self.dropout(F.softmax(torch.matmul(fx, torch.transpose(fp, 1, 2)) / d, dim=2))  # head x Nx x Np
        fa = torch.cat([torch.matmul(w, fp), fx], dim=2)  # head x Nx x 2*fatt
        fa = torch.stack([F.relu(getattr(self, f'fc{i}')(fa[i])) for i in range(self.num_heads)])  # head x Nx x fatt
        fa = torch.transpose(fa, 0, 1).reshape(Nx, -1)  # Nx x fdim
        fx = F.relu(fx_in + self.fc(fa))  # Nx x fdim
        w = torch.transpose(w, 0, 1)  # Nx x head x Np

        return fx, w
    
