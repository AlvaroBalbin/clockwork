"""
Open the trained model and show HOW it generalised.

    python src/grokking/analyze.py

Two figures land in assets_dir:

  grokking_curve.png  train accuracy hits 100% early, test accuracy jumps late.

  fourier_embed.png   the number-embedding matrix, viewed in Fourier space, is
                      almost empty except for a few spikes. That sparsity is the
                      fingerprint of the algorithm the model found: it stores
                      each number n as a handful of waves cos(2*pi*k*n/p), and
                      adds them with the identity
                          cos(a)cos(b) - sin(a)sin(b) = cos(a+b).

Also writes summary.json: the step it memorised, the step it grokked, and the
key frequencies it settled on.
"""
import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from model import Grok

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
runs = os.path.join(ROOT, cfg["runs_dir"])
assets = os.path.join(ROOT, cfg["assets_dir"])
os.makedirs(assets, exist_ok=True)
p = cfg["p"]

# ---- figure 1: the grokking curve ---------------------------------------
hist = json.load(open(os.path.join(runs, "metrics.json")))
steps = [h["step"] for h in hist]
plt.figure(figsize=(8, 5))
plt.plot(steps, [h["train_acc"] for h in hist], label="train acc", lw=2)
plt.plot(steps, [h["test_acc"] for h in hist], label="test acc", lw=2)
plt.xscale("log")
plt.xlabel("optimizer step (log scale)"); plt.ylabel("accuracy")
plt.title("Grokking: memorise early, generalise late")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(assets, "grokking_curve.png"), dpi=130)
print("wrote grokking_curve.png")

memorised_at = next((h["step"] for h in hist if h["train_acc"] > 0.99), None)
grokked_at = next((h["step"] for h in hist if h["test_acc"] > 0.9), None)

# ---- figure 2: Fourier structure of the number embeddings ----------------
model = Grok(cfg)
model.load_state_dict(torch.load(os.path.join(runs, "model.pt"), map_location="cpu"))
WE = model.embed.weight.detach().numpy()[:p]      # number tokens only, [p, d]

# take the DFT down the token axis; the "power" at frequency k is how much of
# the embedding lives on the wave of that frequency. a random matrix would be
# flat here; a grokked one is spiky.
spectrum = np.linalg.norm(np.abs(np.fft.fft(WE, axis=0)), axis=1)
half = spectrum[: p // 2 + 1].copy()
half[0] = 0.0                                     # ignore the constant (DC) term
key = sorted(int(k) for k in np.argsort(half)[::-1][:6] if half[k] > 0.2 * half.max())

plt.figure(figsize=(8, 5))
plt.bar(range(len(half)), half, color="#444")
for k in key:
    plt.bar(k, half[k], color="#d1495b")
plt.xlabel("Fourier frequency k"); plt.ylabel("power in the embedding")
plt.title(f"The embedding is sparse in Fourier space (key frequencies: {key})")
plt.tight_layout()
plt.savefig(os.path.join(assets, "fourier_embed.png"), dpi=130)
print("wrote fourier_embed.png")

summary = dict(memorised_at=memorised_at, grokked_at=grokked_at,
               final_test_acc=round(hist[-1]["test_acc"], 4),
               key_frequencies=key,
               note=f"{len(key)} of {p // 2} frequencies carry the signal")
json.dump(summary, open(os.path.join(runs, "summary.json"), "w"), indent=2)
print("\n" + json.dumps(summary, indent=2))
