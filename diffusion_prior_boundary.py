# file: diffusion_prior_boundary.py
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

# ---- tiny time-embed MLP score model for latent z ----
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
        # z:[B,D], t:[B] in [0,1]
        t = t.view(-1,1)
        te = self.fc_t(t)
        x = torch.cat([z, te], dim=1)
        return self.net(x)

# Denoising Score Matching loss for Gaussian noise level sigma(t)
def sigma_from_t(t, sigma_min=0.01, sigma_max=0.5):
    # exponential schedule
    return sigma_min * (sigma_max/sigma_min)**t

def dsm_loss(score_net, z, t):
    # sample noise and teach score = grad log p(z_sigma)
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
    return (m_true - m_2nd)  # >0 id-side, =0 boundary, <0 flipped

@dataclass
class PriorConfig:
    iters: int = 30
    step_margin: float = 0.5   # toward boundary
    step_prior: float = 0.5    # toward data manifold
    tau: float = 0.1
    sigma_t: float = 0.5       # score evaluated at noise level t* (0..1)
    device: str = "cuda"

@torch.no_grad()
def project_to_boundary(z0, y, W, score_net, cfg: PriorConfig):
    """
    Alternating: one step that reduces margin, one step that denoises with score prior.
    """
    z = F.normalize(z0.to(cfg.device), dim=1).clone()
    y = y.to(cfg.device); W = W.to(cfg.device)
    for k in range(cfg.iters):
        # (1) margin step: gradient descent on margin (push toward 0)
        z.requires_grad_(True)
        m = margin_fn(z, y, W, tau=cfg.tau).mean()
        loss = m   # minimize positive margin
        grad = torch.autograd.grad(loss, z)[0]
        with torch.no_grad():
            z = z - cfg.step_margin * F.normalize(grad, dim=1)
            z = F.normalize(z, dim=1)

        # (2) prior step: score ascent (toward higher data density)
        with torch.no_grad():
            t = torch.full((z.size(0),), cfg.sigma_t, device=z.device).clamp_(0,1)
            s = score_net(z, t)                # ≈ ∇_z log p(z) at ~sigma_t
            z = z + cfg.step_prior * F.normalize(s, dim=1)
            z = F.normalize(z, dim=1)

    return z

# ------------------ tiny train loop (latent only) ------------------
def train_score_latent(z_train, epochs=5, batch=4096, lr=1e-3, device="cuda"):
    model = TimeMLP(z_train.size(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    z_train = z_train.to(device)
    n = z_train.size(0)

    model.train()
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

if __name__ == "__main__":
    # demo with random latents
    D, C, N = 256, 100, 50000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    z_train = F.normalize(torch.randn(N, D), dim=1)
    score_net = train_score_latent(z_train, epochs=3, batch=8192, lr=5e-4, device=device)

    # pretend eval set
    z = F.normalize(torch.randn(4096, D), dim=1)
    y = torch.randint(0, C, (z.size(0),))
    W = F.normalize(torch.randn(C, D), dim=1)

    cfg = PriorConfig(iters=20, step_margin=0.4, step_prior=0.4, tau=0.1, sigma_t=0.6, device=device)
    z_bood = project_to_boundary(z, y, W, score_net, cfg)
    print("BOOD latents:", z_bood.shape)
