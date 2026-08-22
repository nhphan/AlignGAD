"""
AlignGAD reference implementation.

This file is intended for a public research repository. It keeps the model
structure and training/evaluation pipeline explicit, while leaving dataset
loading to the user. To run it on a new dataset, implement `load_graph(name)`.

Expected graph format from `load_graph(name)`:
    A: scipy sparse matrix or numpy array, shape [num_nodes, num_nodes]
    X: scipy sparse matrix or numpy array, shape [num_nodes, num_features]
    y: numpy array, shape [num_nodes], binary labels. Use None for unlabeled data.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class AlignGADConfig:
    seed: int = 42
    epochs: int = 50
    d_prime: int = 16
    hidden_dim: int = 128
    latent_dim: int = 64
    lr: float = 7e-4
    weight_decay: float = 3e-4
    grad_clip: float = 1.0
    alpha: float = 2.0
    beta: float = 0.15
    sup_weight: float = 1.2
    rank_weight: float = 0.15
    cluster_ratios: tuple[float, ...] = (1.0, 0.5, 0.25)
    view_agg_weights: tuple[float, ...] = (0.55, 0.30, 0.15)
    kmeans_max_iter: int = 40
    kmeans_batch_size: int = 8192


CFG = AlignGADConfig()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def to_csr(x) -> sp.csr_matrix:
    if sp.issparse(x):
        return x.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(x, dtype=np.float32))


def make_graph(A) -> sp.csr_matrix:
    A = to_csr(A)
    A = A.maximum(A.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A.astype(np.float32)


def row_norm(A) -> sp.csr_matrix:
    A = to_csr(A)
    deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float32)
    inv = np.zeros_like(deg)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    return sp.diags(inv).dot(A).tocsr().astype(np.float32)


def sym_norm(A) -> sp.coo_matrix:
    A = make_graph(A)
    A_hat = A + sp.eye(A.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(A_hat.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_sqrt = np.zeros_like(deg)
    inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    return sp.diags(inv_sqrt).dot(A_hat).dot(sp.diags(inv_sqrt)).tocoo().astype(np.float32)


def scipy_to_torch_sparse(A, device: torch.device) -> torch.Tensor:
    A = A.tocoo().astype(np.float32)
    idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    val = torch.from_numpy(A.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, torch.Size(A.shape), device=device).coalesce()


def zscore(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return ((X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + eps)).astype(np.float32)


def minmax_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + eps)).astype(np.float32)


def project_features(X, d_prime: int, seed: int) -> np.ndarray:
    X = to_csr(X)
    n, d = X.shape
    if d_prime < min(n, d):
        svd = TruncatedSVD(n_components=d_prime, random_state=seed, n_iter=7)
        out = svd.fit_transform(X).astype(np.float32)
    else:
        dense = X.toarray().astype(np.float32)
        if dense.shape[1] < d_prime:
            out = np.pad(dense, ((0, 0), (0, d_prime - dense.shape[1])), mode="constant")
        else:
            out = dense[:, :d_prime]
    return zscore(out)


def graph_signal_unification(X_proj: np.ndarray, A) -> np.ndarray:
    A = make_graph(A)
    P = row_norm(A)
    X0 = zscore(X_proj)
    low1 = P @ X0
    low2 = P @ low1
    high1 = X0 - low1
    high2 = X0 - low2
    mid = low1 - low2
    X_uni = 0.75 * zscore(low1) + 0.35 * zscore(mid) + 0.90 * zscore(high1) + 0.45 * zscore(high2)
    return zscore(X_uni)


def unify_graph(A, X, cfg: AlignGADConfig) -> tuple[sp.csr_matrix, np.ndarray]:
    A = make_graph(A)
    Xp = project_features(X, cfg.d_prime, cfg.seed)
    Xu = graph_signal_unification(Xp, A)
    return A, Xu


def cluster_labels(X: np.ndarray, k: int, seed: int, cfg: AlignGADConfig) -> np.ndarray:
    n = X.shape[0]
    k = int(max(1, min(k, n)))
    if k == n:
        return np.arange(n, dtype=np.int64)
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        n_init=1,
        max_iter=cfg.kmeans_max_iter,
        batch_size=min(max(cfg.kmeans_batch_size, 2 * k), n),
        reassignment_ratio=0.0,
        init="k-means++",
    )
    labels = km.fit_predict(X).astype(np.int64)
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)


def coarsen(A, X: np.ndarray, labels: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    n = X.shape[0]
    k = int(labels.max()) + 1
    P = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), labels)), shape=(n, k), dtype=np.float32)
    cnt = np.asarray(P.sum(axis=0)).reshape(-1).astype(np.float32)
    cnt[cnt == 0] = 1.0
    Xc = np.asarray((P.T @ X) / cnt[:, None], dtype=np.float32)
    Ac = (P.T @ A @ P).tocsr()
    Ac.setdiag(0)
    Ac.data[:] = 1.0
    Ac.eliminate_zeros()
    return make_graph(Ac), Xc


def cluster_targets(y_original: np.ndarray, original_to_cluster: np.ndarray) -> np.ndarray:
    k = int(np.max(original_to_cluster)) + 1
    yv = np.zeros(k, dtype=np.float32)
    np.maximum.at(yv, original_to_cluster, y_original.astype(np.float32))
    return yv


def build_views(A, X: np.ndarray, y: np.ndarray | None, cfg: AlignGADConfig) -> list[dict]:
    views = []
    n0 = X.shape[0]
    views.append(
        {
            "A": A,
            "X": X.astype(np.float32),
            "original_to_view": np.arange(n0, dtype=np.int64),
            "y_view": y.astype(np.float32) if y is not None else None,
        }
    )
    prev_A, prev_X = A, X.astype(np.float32)
    prev_original_to_view = np.arange(n0, dtype=np.int64)
    for level, ratio in enumerate(cfg.cluster_ratios[1:], start=1):
        target_k = max(1, min(int(n0 * ratio), prev_X.shape[0]))
        labels_prev = cluster_labels(prev_X, target_k, cfg.seed + level, cfg)
        A_c, X_c = coarsen(prev_A, prev_X, labels_prev)
        original_to_cluster = labels_prev[prev_original_to_view]
        views.append(
            {
                "A": A_c,
                "X": X_c,
                "original_to_view": original_to_cluster.astype(np.int64),
                "y_view": cluster_targets(y, original_to_cluster) if y is not None else None,
            }
        )
        prev_A, prev_X = A_c, X_c
        prev_original_to_view = original_to_cluster
    return views


def prepare_view(view: dict, device: torch.device) -> dict:
    return {
        "X": torch.from_numpy(view["X"]).float().to(device),
        "A_norm": scipy_to_torch_sparse(sym_norm(view["A"]), device=device),
        "A_rw": row_norm(view["A"]),
        "original_to_view": view["original_to_view"],
        "y_view": torch.from_numpy(view["y_view"]).float().to(device) if view["y_view"] is not None else None,
    }


class GCNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.res = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(self.lin(torch.sparse.mm(adj, x)) + self.res(x) + self.bias)


class AlignGADModel(nn.Module):
    """AlignGAD node discrepancy scoring network."""

    def __init__(self, d_in: int = CFG.d_prime, hidden: int = CFG.hidden_dim, latent: int = CFG.latent_dim):
        super().__init__()
        self.enc1 = GCNBlock(d_in, hidden)
        self.enc2 = GCNBlock(hidden, hidden)
        self.enc3 = GCNBlock(hidden, latent)
        self.bridge = nn.Sequential(nn.Linear(latent, latent), nn.ReLU(), nn.Linear(latent, latent))
        self.dec1 = GCNBlock(latent, hidden)
        self.dec2 = GCNBlock(hidden, hidden)
        self.out = nn.Linear(hidden, d_in)
        self.calibrator = nn.Sequential(
            nn.Linear(d_in + latent + d_in + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor, aux_scores: torch.Tensor):
        h = self.enc1(x, adj)
        h = self.enc2(h, adj)
        z = self.enc3(h, adj)
        z = self.bridge(z)
        d = self.dec1(z, adj)
        d = self.dec2(d, adj)
        x_gen = self.out(d)
        diff = torch.abs(x - x_gen)
        cal_in = torch.cat([x, z, diff, aux_scores], dim=1)
        logit = self.calibrator(cal_in).squeeze(-1)
        return x_gen, z, logit


def auxiliary_scores_np(X: np.ndarray, A) -> np.ndarray:
    P = row_norm(A)
    neigh = np.asarray(P @ X, dtype=np.float32)
    x_t = torch.from_numpy(X).float()
    n_t = torch.from_numpy(neigh).float()
    affinity = (1.0 - F.cosine_similarity(x_t, n_t, dim=1, eps=1e-8)).numpy()
    high = np.linalg.norm(X - neigh, axis=1)
    deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float32)
    deg_score = np.abs(zscore(np.log1p(deg)[:, None]).reshape(-1))
    return np.vstack([minmax_np(affinity), minmax_np(high), minmax_np(deg_score)]).T.astype(np.float32)


def attach_aux(view: dict, device: torch.device) -> dict:
    aux = auxiliary_scores_np(view["X"].detach().cpu().numpy(), view["A_rw"])
    view["aux"] = torch.from_numpy(aux).float().to(device)
    return view


def view_loss(model: AlignGADModel, view: dict, cfg: AlignGADConfig, device: torch.device) -> torch.Tensor:
    x = view["X"]
    x_gen, z, logit = model(x, view["A_norm"], view["aux"])
    disc = (1.0 - F.cosine_similarity(x, x_gen, dim=1, eps=1e-8)).clamp_min(0.0)
    rec_loss = disc.pow(cfg.alpha).mean()
    var_loss = z.var(dim=0, unbiased=False).mean()
    loss = rec_loss + cfg.beta * var_loss
    if view["y_view"] is not None:
        y = view["y_view"]
        pos = y.sum().clamp_min(1.0)
        neg = (1.0 - y).sum().clamp_min(1.0)
        pos_weight = (neg / pos).clamp(max=30.0)
        bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
        pos_idx = torch.where(y > 0.5)[0]
        neg_idx = torch.where(y <= 0.5)[0]
        if len(pos_idx) > 0 and len(neg_idx) > 0:
            k = min(512, len(pos_idx), len(neg_idx))
            rank = F.softplus(logit[neg_idx[:k]] - logit[pos_idx[:k]] + 0.2).mean()
        else:
            rank = torch.tensor(0.0, device=device)
        loss = loss + cfg.sup_weight * bce + cfg.rank_weight * rank
    return loss


@torch.no_grad()
def score_view(model: AlignGADModel, view: dict) -> np.ndarray:
    x = view["X"]
    x_gen, _, logit = model(x, view["A_norm"], view["aux"])
    disc = (1.0 - F.cosine_similarity(x, x_gen, dim=1, eps=1e-8)).detach().cpu().numpy()
    prob = torch.sigmoid(logit).detach().cpu().numpy()
    aux = view["aux"].detach().cpu().numpy()
    score = 0.45 * minmax_np(prob) + 0.30 * minmax_np(disc) + 0.15 * aux[:, 0] + 0.07 * aux[:, 1] + 0.03 * aux[:, 2]
    return score.astype(np.float32)


@torch.no_grad()
def score_graph(model: AlignGADModel, views: list[dict], n_original: int, cfg: AlignGADConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    mapped = []
    for level, view in enumerate(views):
        sv = score_view(model, view)
        mapped_score = sv if level == 0 else sv[view["original_to_view"]]
        mapped.append(minmax_np(mapped_score))
    stacked = np.vstack(mapped)
    weighted = np.zeros(n_original, dtype=np.float32)
    for w, s in zip(cfg.view_agg_weights, mapped):
        weighted += float(w) * s
    max_score = stacked.max(axis=0)
    final = 0.60 * minmax_np(weighted) + 0.40 * minmax_np(max_score)
    return final.astype(np.float32), mapped


def load_graph(name: str):
    """
    Replace this function with your own dataset loader.

    Example:
        from scipy.io import loadmat
        data = loadmat(f"Dataset/{name}.mat")
        A = data["Network"]
        X = data["Attributes"]
        y = data.get("Label", None)
        return A, X, None if y is None else y.reshape(-1)
    """
    raise NotImplementedError("Please implement load_graph(name) for your dataset format.")


def prepare_dataset(name: str, loader: Callable[[str], tuple], cfg: AlignGADConfig, device: torch.device, with_labels: bool) -> list[dict]:
    A_raw, X_raw, y_raw = loader(name)
    A, X = unify_graph(A_raw, X_raw, cfg)
    y = None if y_raw is None or not with_labels else np.asarray(y_raw, dtype=np.int64).reshape(-1)
    views = build_views(A, X, y, cfg)
    return [attach_aux(prepare_view(v, device), device) for v in views]


def train_aligngad(source_names: Iterable[str], loader: Callable[[str], tuple], cfg: AlignGADConfig = CFG) -> AlignGADModel:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_views = [prepare_dataset(name, loader, cfg, device, with_labels=True) for name in source_names]
    model = AlignGADModel(cfg.d_prime, cfg.hidden_dim, cfg.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        count = 0
        for graph_views in source_views:
            for weight, view in zip(cfg.view_agg_weights, graph_views):
                total = total + view_loss(model, view, cfg, device) * float(weight)
                count += 1
        total = total / max(1, count)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == cfg.epochs:
            print(f"Epoch {epoch:03d}/{cfg.epochs:03d} | loss={float(total.detach().cpu()):.6f}")
    return model


def evaluate_aligngad(model: AlignGADModel, target_names: Iterable[str], loader: Callable[[str], tuple], cfg: AlignGADConfig = CFG) -> pd.DataFrame:
    device = next(model.parameters()).device
    rows = []
    for name in target_names:
        t0 = time.time()
        A_raw, X_raw, y_raw = loader(name)
        A, X = unify_graph(A_raw, X_raw, cfg)
        views = [attach_aux(prepare_view(v, device), device) for v in build_views(A, X, None, cfg)]
        final, _ = score_graph(model, views, X.shape[0], cfg)
        row = {"dataset": name, "n": int(X.shape[0]), "time_sec": time.time() - t0}
        if y_raw is not None:
            y = np.asarray(y_raw, dtype=np.int64).reshape(-1)
            row.update({"auroc": roc_auc_score(y, final), "auprc": average_precision_score(y, final)})
        rows.append(row)
        print(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlignGAD reference runner. Implement load_graph() before running.")
    parser.add_argument("--sources", nargs="+", required=True, help="Source/train graph names.")
    parser.add_argument("--targets", nargs="+", required=True, help="Target/test graph names.")
    parser.add_argument("--results", type=Path, default=Path("results/aligngad_results.csv"))
    args = parser.parse_args()

    model = train_aligngad(args.sources, load_graph, CFG)
    results = evaluate_aligngad(model, args.targets, load_graph, CFG)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results, index=False)
    print(f"Saved results to {args.results}")


if __name__ == "__main__":
    main()
