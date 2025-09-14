# BOOD Boundary Sampling Scripts

This Markdown file contains **two simple scripts** and **key instructions** to run BOOD (Boundary OOD) experiments in your AI IDE (e.g., Cursor).

---

## A) Latent Noise Smoothing (no training)

File: `smoothing_latent.py`

```python
import math, torch, torch.nn.functional as F
from dataclasses import dataclass

@torch.no_grad()
def cosine_logits(z, W, tau=0.1):
    z = F.normalize(z, p=2, dim=1)
    W = F.normalize(W, p=2, dim=1)
    return (z @ W.t()) / tau

def energy_score(logits):
    return -torch.logsumexp(logits, dim=1)

def msp_score(logits):
    return torch.softmax(logits, dim=1).amax(dim=1).values

@dataclass
class SmoothingConfig:
    sigmas: torch.Tensor
    mc: int = 32
    tau: float = 0.1
    device: str = "cuda"

@torch.no_grad()
def estimate_boundary_sigma(z, y, W, cfg: SmoothingConfig):
    B, D = z.shape
    sigma_hat = torch.full((B,), float("inf"), device=cfg.device)
    flip_sigma_idx = torch.full((B,), -1, device=cfg.device, dtype=torch.long)
    base_pred = cosine_logits(z, W, tau=cfg.tau).argmax(1)
    still_same = torch.ones(B, dtype=torch.bool, device=cfg.device)
    for si, sigma in enumerate(cfg.sigmas):
        if not still_same.any(): break
        z_rep = z[still_same].unsqueeze(1).repeat(1, cfg.mc, 1)
        noise = torch.randn_like(z_rep) * sigma
        z_noisy = (z_rep + noise).view(-1, z.shape[1])
        logits = cosine_logits(z_noisy, W, tau=cfg.tau)
        pred = logits.argmax(1).view(-1, cfg.mc)
        maj, _ = torch.mode(pred, dim=1)
        idxs = torch.nonzero(still_same).squeeze(1)
        flipped = (maj != base_pred[idxs])
        sigma_hat[idxs[flipped]] = float(sigma)
        flip_sigma_idx[idxs[flipped]] = si
        still_same[idxs[flipped]] = False
    return sigma_hat, flip_sigma_idx

@torch.no_grad()
def sample_bood_via_smoothing(z, y, W, cfg: SmoothingConfig, target_per_class=20):
    z = z.to(cfg.device); y = y.to(cfg.device); W = W.to(cfg.device)
    B, D = z.shape
    picked_mask = torch.zeros(B, dtype=torch.bool, device=cfg.device)
    base_pred = cosine_logits(z, W, tau=cfg.tau).argmax(1)
    per_class_counts = torch.zeros(W.size(0), dtype=torch.long, device=cfg.device)
    for sigma in cfg.sigmas:
        eps = torch.randn_like(z) * sigma
        z_noisy = z + eps
        pred = cosine_logits(z_noisy, W, tau=cfg.tau).argmax(1)
        flipped = (pred != base_pred) & (~picked_mask)
        cand = torch.nonzero(flipped).squeeze(1)
        for i in cand.tolist():
            t = int(y[i].item())
            if per_class_counts[t] < target_per_class:
                per_class_counts[t] += 1
                picked_mask[i] = True
        if (per_class_counts >= target_per_class).all():
            break
    return z[picked_mask], y[picked_mask], picked_mask
```

Usage:
```python
cfg = SmoothingConfig(sigmas=torch.linspace(0.0, 0.6, 13), mc=32, tau=0.1, device="cuda")
sigma_hat, idx = estimate_boundary_sigma(z, y, W, cfg)
Z_ood, y_src, mask = sample_bood_via_smoothing(z, y, W, cfg, target_per_class=10)
```

---

## B) Diffusion-Prior Projected Boundary (light training)

File: `diffusion_prior_boundary.py`

```python
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

class TimeMLP(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.fc_t = nn.Sequential(nn.Linear(1, hidden//2), nn.SiLU(), nn.Linear(hidden//2, hidden//2))
        self.net  = nn.Sequential(
            nn.Linear(dim + hidden//2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )
    def forward(self, z, t):
        t = t.view(-1,1)
        te = self.fc_t(t)
        x = torch.cat([z, te], dim=1)
        return self.net(x)

def sigma_from_t(t, sigma_min=0.01, sigma_max=0.5):
    return sigma_min * (sigma_max/sigma_min)**t

def dsm_loss(score_net, z, t):
    with torch.no_grad():
        sigma = sigma_from_t(t).view(-1,1)
        noise = torch.randn_like(z) * sigma
        z_noisy = z + noise
        target = -noise / (sigma**2 + 1e-12)
    pred = score_net(z_noisy, t)
    return F.mse_loss(pred, target)

@torch.no_grad()
def cosine_logits(z, W, tau=0.1):
    z = F.normalize(z, dim=1); W = F.normalize(W, dim=1)
    return (z @ W.t()) / tau

def margin_fn(z, y, W, tau=0.1):
    logits = cosine_logits(z, W, tau)
    top2 = torch.topk(logits, 2, dim=1)
    m_true = logits[torch.arange(z.size(0)), y]
    m_2nd  = torch.where(top2.indices[:,0]==y, top2.values[:,1], top2.values[:,0])
    return (m_true - m_2nd)

@dataclass
class PriorConfig:
    iters: int = 30
    step_margin: float = 0.5
    step_prior: float = 0.5
    tau: float = 0.1
    sigma_t: float = 0.5
    device: str = "cuda"

@torch.no_grad()
def project_to_boundary(z0, y, W, score_net, cfg: PriorConfig):
    z = F.normalize(z0.to(cfg.device), dim=1).clone()
    y = y.to(cfg.device); W = W.to(cfg.device)
    for k in range(cfg.iters):
        z.requires_grad_(True)
        m = margin_fn(z, y, W, tau=cfg.tau).mean()
        grad = torch.autograd.grad(m, z)[0]
        with torch.no_grad():
            z = z - cfg.step_margin * F.normalize(grad, dim=1)
            z = F.normalize(z, dim=1)
        with torch.no_grad():
            t = torch.full((z.size(0),), cfg.sigma_t, device=z.device).clamp_(0,1)
            s = score_net(z, t)
            z = z + cfg.step_prior * F.normalize(s, dim=1)
            z = F.normalize(z, dim=1)
    return z

def train_score_latent(z_train, epochs=5, batch=4096, lr=1e-3, device="cuda"):
    model = TimeMLP(z_train.size(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    z_train = z_train.to(device)
    n = z_train.size(0)
    for ep in range(1, epochs+1):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            z = z_train[idx]
            t = torch.rand(z.size(0), device=device)
            loss = dsm_loss(model, z, t)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        print(f"[score-train] epoch {ep} loss {loss.item():.4f}")
    return model
```

Usage:
```python
score_net = train_score_latent(z_train, epochs=3, batch=8192, lr=5e-4)
cfg = PriorConfig(iters=20, step_margin=0.4, step_prior=0.4, tau=0.1, sigma_t=0.6)
z_bood = project_to_boundary(z_eval, y_eval, W, score_net, cfg)
```

---

## Evaluator Stub

File: `eval_bood_stub.py`

```python
import torch
from smoothing_latent import cosine_logits, energy_score, msp_score

@torch.no_grad()
def eval_ood_scores(z_id, z_ood, W, tau=0.1):
    lid = cosine_logits(z_id, W, tau)
    lood = cosine_logits(z_ood, W, tau)
    e_id, e_ood = energy_score(lid), energy_score(lood)
    m_id, m_ood = msp_score(lid), msp_score(lood)
    thr = torch.quantile(e_id, 0.95)
    fpr95 = (e_ood <= thr).float().mean().item()
    return {"E_id_mean": e_id.mean().item(),
            "E_ood_mean": e_ood.mean().item(),
            "MSP_id_mean": m_id.mean().item(),
            "MSP_ood_mean": m_ood.mean().item(),
            "FPR95_energy": fpr95}
```

---

## Quick Start Instructions

1. **Prepare cached features**:  
   - `z_train.pt` (ID train features)  
   - `z_eval.pt` (eval features)  
   - `y_eval.pt` (labels)  
   - `W.pt` (classifier weights)

2. **Route A (Noise Smoothing):**
   - Load `z_eval`, `y_eval`, `W`
   - Run `estimate_boundary_sigma` & `sample_bood_via_smoothing`
   - Evaluate scores with `eval_bood_stub`

3. **Route B (Diffusion Prior):**
   - Train score net with `z_train`
   - Use `project_to_boundary` on eval batch
   - Evaluate scores with `eval_bood_stub`

4. **Metrics:** log AUROC/FPR95, energy/MSP monotonicity vs sigma or iteration.

---

Happy experimenting 🚀
