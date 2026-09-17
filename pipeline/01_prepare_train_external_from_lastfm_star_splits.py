#!/usr/bin/env python3
"""01 — Build TRAIN_EXTERNAL from the frozen Last-FM* splits.

What this script does
---------------------
Loads ``model_train.txt`` and ``valid.txt`` and writes their union
(TRAIN_EXTERNAL): the only interaction population allowed for neighbours
and φ tables on extra.

It does **not** read ``test.txt`` except to count how many users will be
scored later. No ranking happens here.

This is the post-competition extra protocol: statistics fitted as if
validation were already observed, then the sealed test is scored once.

Inputs
------
- ``data/splits/model_train.txt``
- ``data/splits/valid.txt``
- ``data/splits/test.txt`` (user ids / positives counted only)

Outputs
-------
- ``expected/train_external_meta.json`` — interaction count, n_eval_users
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.splits import dict_union, load_user_sets  # noqa: E402

N_USERS = 23566
N_ITEMS = 48123


def main() -> None:
    splits = ROOT / "data" / "splits"
    model_train = load_user_sets(splits / "model_train.txt")
    valid = load_user_sets(splits / "valid.txt")
    test = load_user_sets(splits / "test.txt")
    train_external = dict_union(model_train, valid)
    n_int = sum(len(v) for v in train_external.values())
    n_eval = sum(1 for u, pos in test.items() if pos)
    payload = {
        "protocol": "TRAIN_EXTERNAL = model_train ∪ valid",
        "n_users_matrix": N_USERS,
        "n_items": N_ITEMS,
        "n_train_external_interactions": int(n_int),
        "n_eval_users_with_test_positive": int(n_eval),
        "official_target_interactions": 1235905,
        "official_target_eval_users": 23529,
        "K_USER": 200,
        "TAU": 1.0,
        "no_learned_parameters": True,
    }
    out = ROOT / "expected" / "train_external_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if n_int != 1235905:
        raise SystemExit(f"TRAIN_EXTERNAL size {n_int} != 1235905 — split hash mismatch")
    if n_eval != 23529:
        raise SystemExit(f"eval users {n_eval} != 23529")


if __name__ == "__main__":
    main()
