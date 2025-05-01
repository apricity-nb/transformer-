from torch import nn
import torch.nn.functional as F
from attention import MultiHeadAttention
from layernorm import LayerNorm
from embedding import TransformerEmbedding


class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,dropout= 0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(d_model,hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,dropout = 0.1):
        super(EncoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model,n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model,ffn_hidden,dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        residual = x
        x = self.attention(x,x,x,mask)
        x = self.dropout1(x)
        x = self.norm1(x +  residual)
        residual = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x +  residual)
        return x

class Encoder(nn.Module):
    def __init__(self,enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,dropout=0.1,device="cuda"):
        super(Encoder,self).__init__()
        self.embedding = TransformerEmbedding(enc_voc_size,max_len,d_model,dropout=dropout,device=device)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model,ffn_hidden,n_head,dropout)
                for _ in range(n_layer)
            ]
        )

    def forward(self,x,s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x,s_mask)
        return x
