"""Load Last-FM* user–item sets (one user per line: user_id item_id ...)."""

from __future__ import annotations

from pathlib import Path


def load_user_sets(path: Path) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            u = int(parts[0])
            out[u] = {int(x) for x in parts[1:]}
    return out


def dict_union(a: dict[int, set[int]], b: dict[int, set[int]]) -> dict[int, set[int]]:
    keys = sorted(set(a) | set(b))
    return {u: set(a.get(u, set())) | set(b.get(u, set())) for u in keys}
