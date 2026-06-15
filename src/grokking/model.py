"""
A deliberately small transformer: token + positional embeddings, one attention
head block, one MLP, and an unembedding. No LayerNorm anywhere.

The reason it is this stripped down is that we want to read the algorithm back
out of the weights afterwards. LayerNorm and depth would smear the computation
across the network and make the circuit much harder to see. With a single layer
and no normalisation, the number embeddings themselves end up holding the trick
(a few Fourier components), which is exactly what analyze.py goes looking for.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Grok(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        p, d = cfg["p"], cfg["d_model"]
        self.h, self.dh = cfg["n_heads"], cfg["d_head"]
        self.embed = nn.Embedding(p + 1, d)                 # +1 for the '=' token
        self.pos = nn.Parameter(torch.randn(3, d) / math.sqrt(d))
        self.Wq = nn.Linear(d, self.h * self.dh, bias=False)
        self.Wk = nn.Linear(d, self.h * self.dh, bias=False)
        self.Wv = nn.Linear(d, self.h * self.dh, bias=False)
        self.Wo = nn.Linear(self.h * self.dh, d, bias=False)
        self.mlp_in = nn.Linear(d, cfg["d_mlp"])
        self.mlp_out = nn.Linear(cfg["d_mlp"], d)
        self.unembed = nn.Linear(d, p, bias=False)          # logits over 0..p-1

    def forward(self, x):
        b, t = x.shape
        h = self.embed(x) + self.pos
        q = self.Wq(h).view(b, t, self.h, self.dh).transpose(1, 2)
        k = self.Wk(h).view(b, t, self.h, self.dh).transpose(1, 2)
        v = self.Wv(h).view(b, t, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(t, t, device=x.device), 1).bool()
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        o = (att @ v).transpose(1, 2).reshape(b, t, -1)
        h = h + self.Wo(o)
        h = h + self.mlp_out(F.relu(self.mlp_in(h)))
        return self.unembed(h[:, -1])                       # read off at '='
