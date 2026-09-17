"""Publication full-catalogue metrics (same tie-break as the thesis evaluator).

candidates(u) = {0..n_items-1} \\ train_history(u)
rank by descending score, then smaller item id.
NDCG@K / Recall@K / MRR; positives in the train mask stay in the denominator.
"""

from __future__ import annotations

import numpy as np


def dense_topk(scores: np.ndarray, *, k: int, mask_items) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).copy()
    if mask_items:
        for i in mask_items:
            ii = int(i)
            if 0 <= ii < s.size:
                s[ii] = -np.inf
    item_ids = np.arange(s.size, dtype=np.int64)
    order = np.lexsort((item_ids, -s))
    return order[:k]


def metrics_from_topk(top_items, positive_items, *, ks=(5, 10, 20), full_rank_order_for_mrr=None):
    pos = set(int(x) for x in positive_items)
    if not pos:
        nan = float("nan")
        out = {f"Recall@{k}": nan for k in ks}
        out.update({f"NDCG@{k}": nan for k in ks})
        out["MRR"] = nan
        return out
    out = {}
    top_list = [int(x) for x in np.asarray(top_items).tolist()]
    for k in ks:
        topk = top_list[:k]
        hits = sum(1 for i in topk if i in pos)
        out[f"Recall@{k}"] = float(hits / len(pos))
        dcg = 0.0
        for rank, item in enumerate(topk, start=1):
            if item in pos:
                dcg += 1.0 / np.log2(rank + 1)
        ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(k, len(pos)) + 1))
        out[f"NDCG@{k}"] = float(dcg / ideal) if ideal > 0 else 0.0
    order = (
        [int(x) for x in np.asarray(full_rank_order_for_mrr).tolist()]
        if full_rank_order_for_mrr is not None
        else top_list
    )
    mrr = 0.0
    for rank, item in enumerate(order, start=1):
        if item in pos:
            mrr = 1.0 / rank
            break
    out["MRR"] = float(mrr)
    return out


def evaluate_user_dense(scores, *, positive_items, train_items, ks=(5, 10, 20)):
    k_max = max(ks)
    top = dense_topk(scores, k=k_max, mask_items=train_items)
    s = np.asarray(scores, dtype=np.float64).copy()
    train = {int(i) for i in train_items}
    for i in train:
        if 0 <= i < s.size:
            s[i] = -np.inf
    item_ids = np.arange(s.size, dtype=np.int64)
    full_order = np.lexsort((item_ids, -s))
    eligible = np.asarray([int(i) for i in full_order if int(i) not in train], dtype=np.int64)
    return metrics_from_topk(top, positive_items, ks=ks, full_rank_order_for_mrr=eligible)
