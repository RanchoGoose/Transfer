# file: smoothing_latent.py
import math, torch, torch.nn.functional as F
from dataclasses import dataclass

@torch.no_grad()
def cosine_logits(z, W, tau=0.1):
    # z: [B,D], W: [C,D] (class anchors/last-layer weights)
    z = F.normalize(z, p=2, dim=1)
    W = F.normalize(W, p=2, dim=1)
    return (z @ W.t()) / tau

def energy_score(logits):
    # energy = -tau * logsumexp(logits/tau); but logits already divided by tau
    # so energy = -logsumexp(logits)
    return -torch.logsumexp(logits, dim=1)

def msp_score(logits):
    return torch.softmax(logits, dim=1).amax(dim=1).values

@dataclass
class SmoothingConfig:
    sigmas: torch.Tensor      # e.g., torch.linspace(0.0, 0.6, 13)
    mc: int = 32              # MC samples per sigma
    tau: float = 0.1          # cosine temperature
    device: str = "cuda"

@torch.no_grad()
def estimate_boundary_sigma(z, y, W, cfg: SmoothingConfig):
    """
    Returns:
      sigma_hat: [B] minimal sigma that changes argmax (inf if never flips)
      flip_sigma_idx: [B] index in cfg.sigmas (or -1)
      probs_path: optional per-sigma max prob for debugging
    """
    z = z.to(cfg.device)
    y = y.to(cfg.device)
    W = W.to(cfg.device)

    B, D = z.shape
    sigma_hat = torch.full((B,), float("inf"), device=cfg.device)
    flip_sigma_idx = torch.full((B,), -1, device=cfg.device, dtype=torch.long)

    base_pred = cosine_logits(z, W, tau=cfg.tau).argmax(1)
    still_same = torch.ones(B, dtype=torch.bool, device=cfg.device)

    for si, sigma in enumerate(cfg.sigmas):
        if not still_same.any(): break
        # Monte Carlo smoothing
        z_rep = z[still_same].unsqueeze(1).repeat(1, cfg.mc, 1)  # [b', mc, D]
        noise = torch.randn_like(z_rep) * sigma
        z_noisy = (z_rep + noise).view(-1, z.shape[1])           # [b'*mc, D]
        logits = cosine_logits(z_noisy, W, tau=cfg.tau)
        pred = logits.argmax(1).view(-1, cfg.mc)                 # [b', mc]

        # majority vote
        maj, _ = torch.mode(pred, dim=1)                         # [b']
        idxs = torch.nonzero(still_same).squeeze(1)
        flipped = (maj != base_pred[idxs])

        # write results for first flip
        sigma_hat[idxs[flipped]] = float(sigma)
        flip_sigma_idx[idxs[flipped]] = si
        still_same[idxs[flipped]] = False

    return sigma_hat, flip_sigma_idx

@torch.no_grad()
def sample_bood_via_smoothing(z, y, W, cfg: SmoothingConfig, target_per_class=20):
    """
    For each sample, take the *first* sigma that flips prediction and return that noisy z*
    """
    z = z.to(cfg.device); y = y.to(cfg.device); W = W.to(cfg.device)
    B, D = z.shape
    picked_mask = torch.zeros(B, dtype=torch.bool, device=cfg.device)

    # Pre-compute base predictions
    base_pred = cosine_logits(z, W, tau=cfg.tau).argmax(1)

    per_class_counts = torch.zeros(W.size(0), dtype=torch.long, device=cfg.device)

    for sigma in cfg.sigmas:
        # Generate one MC draw (cheapest) and collect first flips
        eps = torch.randn_like(z) * sigma
        z_noisy = z + eps
        pred = cosine_logits(z_noisy, W, tau=cfg.tau).argmax(1)
        flipped = (pred != base_pred) & (~picked_mask)

        # class balancing by source (true label y)
        cand = torch.nonzero(flipped).squeeze(1)
        for i in cand.tolist():
            t = int(y[i].item())
            if per_class_counts[t] < target_per_class:
                per_class_counts[t] += 1
                picked_mask[i] = True

        if (per_class_counts >= target_per_class).all():
            break

    Z_ood = (z + torch.randn_like(z) * cfg.sigmas[min(len(cfg.sigmas)-1, 1)]).clone()  # fallback (small sigma)
    if picked_mask.any():
        # refine: for picked positions, regenerate noise at *their* first flipping sigma by bisection
        Z_ood = z.clone()
        # simple single-step: use current sigma (already flipped)
        Z_ood[picked_mask] = (z[picked_mask] + torch.randn_like(z[picked_mask]) * sigma).clone()

    y_src = y.clone()
    return Z_ood[picked_mask], y_src[picked_mask], picked_mask

if __name__ == "__main__":
    # --- tiny demo (expects you already cached z,y,W) ---
    D, C, B = 256, 200, 4096
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    z = F.normalize(torch.randn(B, D), dim=1)
    y = torch.randint(0, C, (B,))
    W = F.normalize(torch.randn(C, D), dim=1)

    cfg = SmoothingConfig(sigmas=torch.linspace(0.0, 0.6, 13), mc=32, tau=0.1, device=device)
    sigma_hat, idx = estimate_boundary_sigma(z, y, W, cfg)
    print("median sigma_hat:", torch.median(torch.where(torch.isinf(sigma_hat), torch.tensor(1e9, device=device), sigma_hat)).item())

    Z_ood, y_src, mask = sample_bood_via_smoothing(z, y, W, cfg, target_per_class=10)
    print("BOOD samples:", Z_ood.size(0))
