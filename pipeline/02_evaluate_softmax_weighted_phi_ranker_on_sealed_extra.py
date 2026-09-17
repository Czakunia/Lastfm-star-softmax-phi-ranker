#!/usr/bin/env python3
"""02 — Evaluate the softmax-weighted φ ranker on the sealed extra.

What this script does
---------------------
Frozen post-competition method (thesis follow-up, 2nd vs RP3β on extra):

1. Fit the binary user–item matrix R on TRAIN_EXTERNAL only
   (model_train ∪ valid). Never uses test interactions for co-occurrence.
2. For each test user u:
   a. Retrieve C_CF(u): items of the 200 nearest users by cosine of
      binary histories (overlap ≥ 1), excluding H_u.
   b. For each candidate X in C_CF(u), compute Pearson φ(h, X) against
      every history item h ∈ H_u.
   c. Score X as the softmax-weighted mean of those φ values
      (default τ=1; ``--tau 0.5`` is the supported sharper setting).
3. Rank the full catalogue (unretrieved items stay at −∞) and compute
   NDCG@20 / Recall@20 / MRR with the publication evaluator.

No learned weights. k=200 is frozen. τ=1 is the published extra default
(NDCG@20 = 0.215101, n=23529). τ=0.5 is allowed and stays the better
Last-FM* extra (NDCG@20 = 0.2185), still 2nd vs RP3β (~0.232); it does
not replace the frozen τ=1 number.

This run writes ``verify_rerun/sealed_metrics.json`` so the number can be
checked again without touching the frozen copy in ``expected/``.

Inputs
------
- ``data/splits/{model_train,valid,test}.txt``

Outputs
-------
- ``verify_rerun/sealed_metrics.json``
- ``verify_rerun/LIVE.txt``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import evaluate_user_dense  # noqa: E402
from src.phi_ranker import (  # noqa: E402
    K_USER,
    TAU,
    TAU_SHARP,
    build_urm,
    cf_candidates,
    item_user_lists,
    score_phi,
)
from src.splits import dict_union, load_user_sets  # noqa: E402

N_USERS = 23566
N_ITEMS = 48123
FROZEN_NDCG20 = 0.21510094169360347  # τ=1 extra
SHARP_NDCG20 = 0.21854866811888393  # τ=0.5 extra; still 2nd vs RP3β


def live(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    prev = path.read_text() if path.exists() else ""
    path.write_text(prev + line + "\n")


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("NDCG@20", "Recall@20", "MRR")}
    out["n_users"] = float(len(rows))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Sealed extra softmax-weighted φ ranker.")
    p.add_argument(
        "--tau",
        type=float,
        default=TAU,
        help=(
            "softmax temperature (default 1.0 = frozen extra 0.2151). "
            "Pass 0.5 for the sharper Last-FM* extra (0.2185); it stays 2nd vs RP3β "
            "and does not replace the frozen τ=1 number."
        ),
    )
    args = p.parse_args()
    tau = float(args.tau)

    out_dir = ROOT / "verify_rerun"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "LIVE.txt"
    splits = ROOT / "data" / "splits"
    model_train = load_user_sets(splits / "model_train.txt")
    valid = load_user_sets(splits / "valid.txt")
    test = load_user_sets(splits / "test.txt")
    train_external = dict_union(model_train, valid)
    scored = sorted(int(u) for u, pos in test.items() if pos)
    live(
        log,
        f"TRAIN_EXTERNAL interactions={sum(len(v) for v in train_external.values())} "
        f"eval_users={len(scored)} tau={tau}",
    )

    R = build_urm(train_external, N_USERS, N_ITEMS)
    item_degree = np.asarray(R.sum(axis=0)).ravel().astype(np.float64)
    n_v = np.asarray(R.sum(axis=1)).ravel().astype(np.float64)
    N = float(N_USERS)
    p_item = item_degree / N
    den_item = np.sqrt(np.maximum(p_item * (1.0 - p_item), 0.0))
    hists = [np.fromiter((int(i) for i in train_external.get(u, ())), dtype=np.int64) for u in range(N_USERS)]
    avail = n_v >= 1.0
    item_users = item_user_lists(R)
    live(log, "item–item co-occurrence C = R.T @ R …")
    C = (R.T @ R).tocsr()
    C.sort_indices()

    rows = []
    t0 = time.time()
    for j, u in enumerate(scored):
        hu = hists[u]
        train = set(int(x) for x in hu.tolist())
        cand = cf_candidates(u, hu, R=R, n_v=n_v, item_users=item_users, avail=avail, hists=hists, k=K_USER)
        sc = score_phi(hu, cand, C=C, p_item=p_item, den_item=den_item, n_users=N, n_items=N_ITEMS, tau=tau)
        pos = set(int(x) for x in test[u])
        m = evaluate_user_dense(sc, positive_items=pos, train_items=train, ks=(5, 10, 20))
        rows.append(
            {
                "NDCG@20": float(m["NDCG@20"]),
                "Recall@20": float(m["Recall@20"]),
                "MRR": float(m["MRR"]),
            }
        )
        if (j + 1) % 2000 == 0 or j == 0:
            nd = float(np.mean([r["NDCG@20"] for r in rows]))
            live(log, f"extra {j+1}/{len(scored)}  running NDCG@20={nd:.6f}")

    overall = mean_metrics(rows)
    ref = FROZEN_NDCG20 if abs(tau - TAU) < 1e-12 else (SHARP_NDCG20 if abs(tau - TAU_SHARP) < 1e-12 else None)
    blob = {
        "method": f"softmax-weighted Pearson φ on UserKNN C_CF (k=200, τ={tau:g})",
        "protocol": "TRAIN_EXTERNAL → sealed test.txt",
        "tau": tau,
        "tau_frozen_default": TAU,
        "tau_sharp_supported": TAU_SHARP,
        "overall": overall,
        "frozen_reference_NDCG@20": FROZEN_NDCG20,
        "sharp_reference_NDCG@20": SHARP_NDCG20,
        "abs_diff_vs_reference": None if ref is None else abs(overall["NDCG@20"] - ref),
        "elapsed_sec": time.time() - t0,
        "official_tables": False,
        "thesis_follow_up": True,
        "note": (
            "τ=1 is the frozen extra (0.2151). τ=0.5 is supported and stays "
            "2nd vs RP3β on Last-FM* extra (0.2185); it does not replace τ=1."
        ),
    }
    (out_dir / "sealed_metrics.json").write_text(json.dumps(blob, indent=2) + "\n")
    delta = blob["abs_diff_vs_reference"]
    live(
        log,
        f"DONE tau={tau:g} NDCG@20={overall['NDCG@20']:.6f}  "
        f"frozen_τ1={FROZEN_NDCG20:.6f}  sharp_τ0.5={SHARP_NDCG20:.6f}"
        + ("" if delta is None else f"  |Δ_vs_ref|={delta:.6g}")
        + f"  n={int(overall['n_users'])}",
    )


if __name__ == "__main__":
    main()
