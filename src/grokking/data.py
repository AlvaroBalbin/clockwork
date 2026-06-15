"""
The dataset is every pair (a, b) with a, b in 0..p-1, and the label is
(a + b) mod p. There is no file to download, the whole thing is generated.

Each example is a length-3 sequence [a, b, =], where = is a special token
(id p) that marks where the answer should be read off. The model sees the
first two tokens and has to predict the sum at the last position.

The split is the important bit: we hand the model only train_frac of the pairs
and hide the rest. Grokking is precisely the model going from "wrong on the
hidden pairs" to "right on the hidden pairs" long after it has nailed the ones
it can see, so we need that held-out set to watch it happen.
"""
import torch


def build(cfg, device):
    p = cfg["p"]
    a = torch.arange(p).repeat_interleave(p)      # 0,0,..,0,1,1,..
    b = torch.arange(p).repeat(p)                 # 0,1,..,p-1,0,1,..
    eq = torch.full_like(a, p)                    # the '=' token
    x = torch.stack([a, b, eq], dim=1).to(device)  # [p*p, 3]
    y = ((a + b) % p).to(device)                  # [p*p]

    # deterministic split so the same seed gives the same held-out pairs
    g = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(p * p, generator=g)
    n_train = int(cfg["train_frac"] * p * p)
    train_idx = perm[:n_train].to(device)
    test_idx = perm[n_train:].to(device)
    return x, y, train_idx, test_idx
