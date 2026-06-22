"""
Quick checks before spending 25 minutes training. Run:

    python notebooks/00_sanity_check.py

It confirms the dataset is shaped right and labelled right, and that one
forward pass through a fresh model produces a loss near ln(p), which is what
you should get from a model that is still guessing uniformly over p classes. If
the starting loss is way off ln(p), something is wired wrong and no amount of
training will save it.
"""
import os, sys, math
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "grokking"))
from data import build
from model import Grok

ROOT = os.path.join(os.path.dirname(__file__), "..")
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
p = cfg["p"]

X, Y, tr, te = build(cfg, "cpu")
assert X.shape == (p * p, 3), X.shape
assert (X[:, 2] == p).all(), "third token should always be the '=' token"
# spot-check a few labels by hand
for i in (0, 1234, p * p - 1):
    a, b = int(X[i, 0]), int(X[i, 1])
    assert int(Y[i]) == (a + b) % p, (a, b, int(Y[i]))
assert len(tr) + len(te) == p * p and len(set(tr.tolist()) & set(te.tolist())) == 0
print(f"data ok: {p*p} pairs, {len(tr)} train / {len(te)} test, no overlap")

torch.manual_seed(0)
model = Grok(cfg)
loss = F.cross_entropy(model(X[:512]), Y[:512]).item()
print(f"fresh-model loss {loss:.3f}   (ln p = {math.log(p):.3f}, so guessing looks right)")
assert abs(loss - math.log(p)) < 1.0, "starting loss is nowhere near uniform, check the model"
print("all good, safe to train")
