import math

import torch
from torch import nn

x = torch.rand(128,32,512)
d_model = 512
n_head = 8

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model,d_model)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,q,k,v,mask=None):
        batch,time,dimention = q.shape
        n_d=self.d_model // self.n_head
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)
        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        score = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(n_d)
        if mask is not None:
            score = score.masked_fill(mask==0,-10000)
        score = torch.matmul(self.softmax(score),v)
        score = score.permute(0,2,1,3).contiguous().view(batch,time,dimention)
        out = self.w_combine(score)
        return out

attention = MultiHeadAttention(d_model,n_head)
out = attention(x,x,x)
print(out)
