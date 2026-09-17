"""UserKNN retrieval (k=200) and softmax-weighted Pearson φ scoring.

No learned parameters. Candidates C_CF(u) are items listened to by the
200 nearest users (cosine of binary histories), excluding the query
user's own TRAIN_EXTERNAL history.

Temperature
-----------
``TAU = 1.0`` is the frozen extra default (NDCG@20 = 0.2151).
``TAU_SHARP = 0.5`` is a supported alternative: same φ and same C_CF,
only softmax(φ/τ) is sharper. On this Last-FM* extra it stays 2nd vs
RP3β (~0.232) at NDCG@20 = 0.2185. It does not replace the frozen τ=1
number. Smaller τ → more mass on the strongest history–candidate φ;
τ → ∞ recovers the uniform mean of φ.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

K_USER = 200
TAU = 1.0
TAU_SHARP = 0.5  # optional; Last-FM* extra NDCG@20 = 0.2185, still 2nd vs RP3β


def build_urm(train: dict[int, set[int]], n_users: int, n_items: int) -> sps.csr_matrix:
    rows, cols, data = [], [], []
    for u, items in train.items():
        for i in items:
            rows.append(int(u))
            cols.append(int(i))
            data.append(1.0)
    return sps.csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float64)


def item_user_lists(R: sps.csr_matrix) -> list[np.ndarray]:
    Rc = R.tocsc()
    return [Rc.indices[slice(Rc.indptr[i], Rc.indptr[i + 1])].astype(np.int64) for i in range(R.shape[1])]


def eligible_neighbors(u: int, hist: np.ndarray, item_users: list[np.ndarray], available: np.ndarray) -> np.ndarray:
    if hist.size == 0:
        return np.zeros(0, dtype=np.int64)
    parts = [item_users[int(i)] for i in hist.tolist()]
    cand = np.unique(np.concatenate(parts))
    cand = cand[cand != int(u)]
    if cand.size == 0:
        return cand
    return cand[available[cand]]


def overlaps_with(R: sps.csr_matrix, user_ids: np.ndarray, hist: np.ndarray) -> np.ndarray:
    if user_ids.size == 0 or hist.size == 0:
        return np.zeros(user_ids.size, dtype=np.float64)
    return np.asarray(R[user_ids][:, hist].sum(axis=1)).ravel().astype(np.float64)


def rank_cos_neighbors(elig: np.ndarray, cosine: np.ndarray, overlap: np.ndarray, k: int = K_USER) -> np.ndarray:
    if elig.size == 0:
        return elig
    order = np.lexsort((elig, -overlap, -cosine))
    return elig[order][: min(int(k), elig.size)]


def softmax_cols(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=0, keepdims=True)
    e = np.exp(logits - m)
    z = e.sum(axis=0, keepdims=True)
    return np.divide(e, z, out=np.zeros_like(e), where=z != 0.0)


def phi_block(C_hx, p_h, p_x, den_h, den_x, n_users: float) -> np.ndarray:
    p_hx = C_hx / float(n_users)
    num = p_hx - p_h[:, None] * p_x[None, :]
    den = den_h[:, None] * den_x[None, :]
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0.0)


def cf_candidates(u: int, hu: np.ndarray, *, R, n_v, item_users, avail, hists, k: int = K_USER) -> np.ndarray:
    train = set(int(x) for x in hu.tolist())
    elig = eligible_neighbors(u, hu, item_users, avail)
    ov = overlaps_with(R, elig, hu)
    keep = ov >= 1.0
    elig, ov = elig[keep], ov[keep]
    if not elig.size:
        return np.zeros(0, dtype=np.int64)
    den = np.sqrt(max(float(hu.size), 1.0) * np.maximum(n_v[elig], 1.0))
    cos = ov / den
    nb = rank_cos_neighbors(elig, cos, ov, k=k)
    if not nb.size:
        return np.zeros(0, dtype=np.int64)
    items = np.unique(np.concatenate([hists[int(v)] for v in nb.tolist()]))
    return np.asarray([int(i) for i in items.tolist() if int(i) not in train], dtype=np.int64)


def score_phi(hu: np.ndarray, cand: np.ndarray, *, C, p_item, den_item, n_users: float, n_items: int, tau: float = TAU) -> np.ndarray:
    """Softmax-weighted φ. Pass ``tau=TAU_SHARP`` (0.5) for the sharper Last-FM* extra."""
    sc = np.full(n_items, -np.inf, dtype=np.float64)
    if hu.size and cand.size:
        C_hx = np.asarray(C[hu][:, cand].todense(), dtype=np.float64)
        phi = phi_block(C_hx, p_item[hu], p_item[cand], den_item[hu], den_item[cand], n_users)
        alpha = softmax_cols(phi / tau)
        sc[cand] = (alpha * phi).sum(axis=0)
    return sc
