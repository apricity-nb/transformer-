import torch
from torch import nn

from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 max_len,
                 n_heads,
                 ffn_hidden,
                 n_layers,
                 drop_prob,
                 device
                 ):
        super(Transformer,self).__init__()
        self.encoder=Encoder(
            enc_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device
        )
        self.decoder=Decoder(
            dec_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = "cuda"

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        q_mask = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, len_q]
        q_mask = q_mask.repeat(1, 1, len_k, 1)  # [batch_size, 1, len_k, len_q]
        k_mask = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(3)  # [batch_size, 1, len_k, 1]
        k_mask = k_mask.repeat(1, 1, 1, len_q)  # [batch_size, 1, len_k, len_q]
        mask = q_mask & k_mask
        return mask

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_pad_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)
        trg_causal_mask = self.make_casual_mask(trg, trg)
        trg_mask = trg_pad_mask & trg_causal_mask
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, enc, trg_mask, src_mask)
        return out

