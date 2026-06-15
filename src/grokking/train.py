"""
Train the model and log the train/test gap the whole way through.

Run it from the repo root:

    python src/grokking/train.py            # 20000 steps, ~25 min on CPU
    python src/grokking/train.py 300        # a quick smoke run

Nothing here is batched: the training set is small enough to do full-batch
gradient descent, which is the setup grokking was first reported in. The only
slightly unusual choices are the very heavy weight decay (in config.yaml) and
the fact that we just keep going long after train accuracy hits 100%. That
patience is the point; the interesting thing happens thousands of steps later.

Writes model.pt and metrics.json into runs_dir.
"""
import os, sys, json, time
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from data import build
from model import Grok

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
if len(sys.argv) > 1:
    cfg["steps"] = int(sys.argv[1])          # let the CLI override for smoke runs

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(cfg["seed"])
runs = os.path.join(ROOT, cfg["runs_dir"])
os.makedirs(runs, exist_ok=True)

X, Y, tr, te = build(cfg, device)
model = Grok(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                        weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]))


@torch.no_grad()
def score(idx):
    logits = model(X[idx])
    loss = F.cross_entropy(logits, Y[idx]).item()
    acc = (logits.argmax(-1) == Y[idx]).float().mean().item()
    return loss, acc


hist, t0 = [], time.time()
for step in range(cfg["steps"] + 1):
    model.train()
    loss = F.cross_entropy(model(X[tr]), Y[tr])
    opt.zero_grad(); loss.backward(); opt.step()
    if step % cfg["log_every"] == 0:
        trl, tra = score(tr)
        tel, tea = score(te)
        hist.append(dict(step=step, train_loss=trl, train_acc=tra,
                         test_loss=tel, test_acc=tea))
        print(f"step {step:6d}  train_acc {tra:.3f}  test_acc {tea:.3f}  "
              f"test_loss {tel:.4f}  ({time.time()-t0:.0f}s)")

json.dump(hist, open(os.path.join(runs, "metrics.json"), "w"))
torch.save(model.state_dict(), os.path.join(runs, "model.pt"))
print(f"\ndone in {time.time()-t0:.0f}s  ->  {cfg['runs_dir']}/model.pt, "
      f"{cfg['runs_dir']}/metrics.json")
