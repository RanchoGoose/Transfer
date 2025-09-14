import os
import math
import pickle
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


try:
    import matplotlib
    matplotlib.use("Agg")  # Safe for headless environments
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None  # type: ignore

try:
    from sklearn.decomposition import PCA  # type: ignore
except Exception:  # pragma: no cover - will fallback to torch
    PCA = None  # type: ignore


Tensor = torch.Tensor


@dataclass
class BoundaryConfig:
    # PGD / Boundary probing
    num_steps: int = 40
    step_size: float = 0.01
    temperature: float = 0.1
    random_start: bool = False
    # OOD collection
    method: str = "boundary_crossing"  # "boundary_crossing" | "threshold_filtering"
    threshold: float = 0.2
    max_ood_per_run: int = 10000
    capture_step_after_cross: int = 1  # like cstep
    # Kappa → boundary/core split
    boundary_rate: float = 0.2  # take lowest-kappa this fraction as boundary per class
    # Visualization
    viz_sample_trajectories: int = 10
    plot_path: Optional[str] = None
    title: str = "Decision Boundary (PCA 2D)"


def _ensure_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    return device


def _maybe_normalize(x: Tensor) -> Tensor:
    return F.normalize(x, p=2, dim=-1)


def cosine_logits(features: Tensor, anchors: Tensor, temperature: float = 0.1) -> Tensor:
    # Efficient cosine via normalized matmul
    features_n = _maybe_normalize(features)
    anchors_n = _maybe_normalize(anchors)
    return (features_n @ anchors_n.t()) / temperature


def compute_anchors(
    features: Tensor,
    labels: Tensor,
    num_classes: int,
    method: str = "mean",
    normalize: bool = True,
) -> Tensor:
    anchors: List[Tensor] = []
    for c in range(num_classes):
        class_mask = labels == c
        if class_mask.sum() == 0:
            # If no samples present for this class, create a zero anchor
            anchors.append(torch.zeros(features.shape[1], device=features.device))
            continue
        feats_c = features[class_mask]
        if method == "mean":
            anchor_c = feats_c.mean(dim=0)
        elif method == "median":
            anchor_c = feats_c.median(dim=0).values
        else:
            raise ValueError(f"Unknown anchor method: {method}")
        anchors.append(anchor_c)
    anchors_t = torch.stack(anchors, dim=0)
    if normalize:
        anchors_t = _maybe_normalize(anchors_t)
    return anchors_t


def default_feature_forward(model: torch.nn.Module, x: Tensor) -> Tensor:
    # Heuristic feature extraction: try common conventions, then fallback to model(x)
    if hasattr(model, "forward_features"):
        return getattr(model, "forward_features")(x)
    if hasattr(model, "extract_features"):
        return getattr(model, "extract_features")(x)
    if hasattr(model, "penultimate"):
        return getattr(model, "penultimate")(x)
    y = model(x)
    if isinstance(y, (tuple, list)) and len(y) >= 2:
        # assume (logits, features)
        return y[1]
    # Fallback: use outputs directly as features
    return y


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: Optional[Union[str, torch.device]] = None,
    feature_forward: Optional[Callable[[torch.nn.Module, Tensor], Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    device = _ensure_device(device)
    model.eval()
    feature_forward = feature_forward or default_feature_forward

    feats: List[Tensor] = []
    lbls: List[Tensor] = []
    for batch in dataloader:
        # Support dataloaders returning (x, y) or (idx, x, y)
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                x, y = batch
            elif len(batch) >= 3:
                x, y = batch[1], batch[2]
            else:
                raise ValueError("Unexpected batch structure from dataloader")
        else:
            raise ValueError("Dataloader must yield tuples/lists")

        x = x.to(device)
        y = y.to(device)

        f = feature_forward(model, x)
        f = f.detach()
        if f.dim() > 2:
            f = f.view(f.size(0), -1)
        feats.append(f)
        lbls.append(y)

    features = torch.cat(feats, dim=0)
    labels = torch.cat(lbls, dim=0).long()
    return features, labels


def pgd_kappa_and_oods(
    features: Tensor,
    labels: Tensor,
    anchors: Tensor,
    cfg: BoundaryConfig,
) -> Tuple[Tensor, Tensor, List[Tensor], List[int]]:
    device = features.device
    x_adv = features.detach().clone()
    if cfg.random_start:
        x_adv = x_adv + 0.001 * torch.randn_like(x_adv, device=device)

    num_samples = x_adv.size(0)
    kappa = torch.zeros(num_samples, device=device)
    collected_oods: List[Tensor] = []
    collected_targets: List[int] = []
    steps_since_cross = torch.full((num_samples,), -1, device=device)
    added_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)

    for step_index in range(cfg.num_steps):
        x_adv.requires_grad_(True)
        logits = cosine_logits(x_adv, anchors, temperature=cfg.temperature)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        # Update kappa (times still correctly predicted)
        kappa = kappa + (pred == labels).float()

        # OOD collection strategies
        if cfg.method == "threshold_filtering":
            max_probs = probs.max(dim=1).values
            take = (max_probs <= cfg.threshold) & (~added_mask)
            if take.any():
                idxs = take.nonzero(as_tuple=False).squeeze(1)
                collected_oods.extend(x_adv[idxs].detach().split(1, dim=0))
                collected_targets.extend(labels[idxs].tolist())
                added_mask[idxs] = True
        elif cfg.method == "boundary_crossing":
            crossed = (pred != labels)
            # count steps after crossing for each sample
            steps_since_cross[crossed & (steps_since_cross < 0)] = 0
            steps_since_cross[crossed & (steps_since_cross >= 0)] += 1

            ready = (
                crossed
                & (steps_since_cross == cfg.capture_step_after_cross)
                & (~added_mask)
            )
            if ready.any():
                idxs = ready.nonzero(as_tuple=False).squeeze(1)
                collected_oods.extend(x_adv[idxs].detach().split(1, dim=0))
                collected_targets.extend(labels[idxs].tolist())
                added_mask[idxs] = True
        else:
            raise ValueError(f"Unknown OOD method: {cfg.method}")

        if len(collected_targets) >= cfg.max_ood_per_run:
            break

        # Compute CE loss and PGD update
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + cfg.step_size * x_adv.grad.sign()
        x_adv = x_adv.detach()

    return x_adv, kappa, collected_oods, collected_targets


def split_boundary_core_per_class(
    kappa: Tensor,
    labels: Tensor,
    num_classes: int,
    rate: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Return per-class indices for boundary (low kappa) and core (others)
    boundary_ids: List[np.ndarray] = []
    core_ids: List[np.ndarray] = []
    kappa_cpu = kappa.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()

    for c in range(num_classes):
        idx_c = np.where(labels_cpu == c)[0]
        if len(idx_c) == 0:
            boundary_ids.append(np.array([], dtype=int))
            core_ids.append(np.array([], dtype=int))
            continue
        kappa_c = kappa_cpu[idx_c]
        order = np.argsort(kappa_c)
        num_boundary = max(1, int(round(len(idx_c) * rate)))
        b_ids = idx_c[order[:num_boundary]]
        c_ids = idx_c[order[num_boundary:]]
        boundary_ids.append(b_ids)
        core_ids.append(c_ids)
    return boundary_ids, core_ids


def _pca_fit_transform(x_np: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if PCA is not None:
        pca = PCA(n_components=n_components)
        z = pca.fit_transform(x_np)
        info = {
            "components": pca.components_.copy(),
            "mean": pca.mean_.copy(),
            "explained_var": pca.explained_variance_.copy(),
        }
        return z, info
    # Torch fallback
    x = torch.from_numpy(x_np)
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    # low-rank PCA
    U, S, V = torch.pca_lowrank(x_centered, q=n_components)
    z = (x_centered @ V[:, :n_components]).numpy()
    info = {"components": V[:, :n_components].t().numpy(), "mean": x_mean.squeeze(0).numpy()}
    return z, info


def _pca_inverse(z2d: np.ndarray, info: Dict[str, np.ndarray]) -> np.ndarray:
    comps = info["components"]  # shape (2, D)
    mean = info["mean"]  # shape (D,)
    # z2d: (N, 2), comps: (2, D)
    x_rec = z2d @ comps + mean  # (N, D)
    return x_rec


def visualize_decision_boundary(
    features: Tensor,
    labels: Tensor,
    anchors: Tensor,
    cfg: BoundaryConfig,
) -> Optional[str]:
    if plt is None:
        return None

    device = features.device
    features_np = features.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    anchors_np = anchors.detach().cpu().numpy()

    z_feat, pinfo = _pca_fit_transform(features_np, n_components=2)
    z_anch, _ = _pca_fit_transform(anchors_np, n_components=2)

    # Grid in 2D PCA space
    x_min, x_max = z_feat[:, 0].min() - 1.0, z_feat[:, 0].max() + 1.0
    y_min, y_max = z_feat[:, 1].min() - 1.0, z_feat[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_hd = _pca_inverse(grid_2d, pinfo)
    grid_hd_t = torch.from_numpy(grid_hd).to(device, dtype=features.dtype)

    with torch.no_grad():
        logits = cosine_logits(grid_hd_t, anchors.to(device), temperature=cfg.temperature)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
    Z = pred.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    # Decision regions
    ax.contourf(xx, yy, Z, alpha=0.2, levels=np.arange(anchors.size(0) + 1) - 0.5, cmap="tab20")
    # Samples
    sc = ax.scatter(z_feat[:, 0], z_feat[:, 1], c=labels_np, cmap="tab20", s=8, alpha=0.7, edgecolors="none")
    # Anchors
    ax.scatter(z_anch[:, 0], z_anch[:, 1], c=np.arange(anchors.size(0)), cmap="tab20", s=80, marker="X", edgecolors="k")

    # Optional: a few PGD trajectories
    if cfg.viz_sample_trajectories > 0:
        num = min(cfg.viz_sample_trajectories, features.size(0))
        sel = torch.randperm(features.size(0))[:num]
        x0 = features[sel]
        y0 = labels[sel]
        anchors_d = anchors.to(device)
        x_adv = x0.clone()
        trj = [x_adv.detach().cpu().numpy()]
        for _ in range(min(cfg.num_steps, 20)):
            x_adv.requires_grad_(True)
            logits = cosine_logits(x_adv, anchors_d, temperature=cfg.temperature)
            loss = F.cross_entropy(logits, y0)
            loss.backward()
            with torch.no_grad():
                x_adv = x_adv + cfg.step_size * x_adv.grad.sign()
            x_adv = x_adv.detach()
            trj.append(x_adv.detach().cpu().numpy())
        trj_np = [t @ np.zeros((t.shape[1], 0)) for t in trj]  # keep shape
        # Project and plot
        for i in range(len(trj)):
            z = (trj[i] - pinfo["mean"]) @ pinfo["components"].T
            if i == 0:
                ax.plot(z[:, 0], z[:, 1], "k.", ms=2, alpha=0.6)
            else:
                ax.plot(z[:, 0], z[:, 1], "k-", lw=0.4, alpha=0.3)

    ax.set_title(cfg.title)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    plt.tight_layout()

    out_path = cfg.plot_path or os.path.join(os.getcwd(), "decision_boundary.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def investigate_decision_boundary(
    model: Optional[torch.nn.Module],
    dataloader: Optional[Iterable],
    num_classes: int,
    device: Optional[Union[str, torch.device]] = None,
    cfg: Optional[BoundaryConfig] = None,
    precomputed_features: Optional[Tuple[Tensor, Tensor]] = None,
    feature_forward: Optional[Callable[[torch.nn.Module, Tensor], Tensor]] = None,
) -> Dict[str, object]:
    """
    Main entry point. If precomputed_features is provided, model/dataloader are optional.

    Returns dict with keys: features, labels, anchors, kappa, boundary_ids, core_ids,
    ood_features, ood_targets, plot_path
    """
    device = _ensure_device(device)
    cfg = cfg or BoundaryConfig()

    if precomputed_features is not None:
        features, labels = precomputed_features
        features = features.to(device)
        labels = labels.to(device).long()
    else:
        if model is None or dataloader is None:
            raise ValueError("Either provide (model and dataloader) or precomputed_features")
        features, labels = extract_features(model, dataloader, device=device, feature_forward=feature_forward)

    if features.dim() > 2:
        features = features.view(features.size(0), -1)

    anchors = compute_anchors(features, labels, num_classes=num_classes, method="mean", normalize=True)

    adv_features, kappa, ood_list, ood_tgts = pgd_kappa_and_oods(features, labels, anchors, cfg)

    boundary_ids, core_ids = split_boundary_core_per_class(kappa, labels, num_classes, rate=cfg.boundary_rate)

    plot_path = visualize_decision_boundary(features, labels, anchors, cfg)

    result: Dict[str, object] = {
        "features": features.detach().cpu(),
        "labels": labels.detach().cpu(),
        "anchors": anchors.detach().cpu(),
        "adv_features": adv_features.detach().cpu(),
        "kappa": kappa.detach().cpu(),
        "boundary_ids": boundary_ids,
        "core_ids": core_ids,
        "ood_features": torch.cat(ood_list, dim=0).cpu() if len(ood_list) > 0 else torch.empty(0),
        "ood_targets": np.array(ood_tgts, dtype=int) if len(ood_tgts) > 0 else np.array([], dtype=int),
        "plot_path": plot_path,
    }
    return result


def save_boundary_core(
    save_dir: str,
    boundary_ids: List[np.ndarray],
    core_ids: List[np.ndarray],
    kappa: Tensor,
    kappa_step_value: Optional[int] = None,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "boundary.bin"), "wb") as f:
        pickle.dump(boundary_ids, f)
    with open(os.path.join(save_dir, "core.bin"), "wb") as f:
        pickle.dump(core_ids, f)
    with open(os.path.join(save_dir, "kappa.bin"), "wb") as f:
        pickle.dump(kappa.detach().cpu().numpy(), f)
    if kappa_step_value is not None:
        kappa_vals = kappa.detach().cpu().numpy()
        idxs = np.where(kappa_vals == kappa_step_value)[0].tolist()
        with open(os.path.join(save_dir, "kappa_step.bin"), "wb") as f:
            pickle.dump(idxs, f)


__all__ = [
    "BoundaryConfig",
    "extract_features",
    "compute_anchors",
    "cosine_logits",
    "pgd_kappa_and_oods",
    "split_boundary_core_per_class",
    "visualize_decision_boundary",
    "investigate_decision_boundary",
    "save_boundary_core",
]


if __name__ == "__main__":
    # Minimal example usage (expects user to integrate with their model/dataloader)
    print(
        "This module provides investigate_decision_boundary().\n"
        "Integrate with your training script by passing your model and dataloader."
    )

