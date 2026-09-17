# Data — softmax-weighted φ ranker

Identical Last-FM* resource as the HGT package.

1. `LastFM_star_IntentAwareRS/` — IntentAwareRS SIGIR 2025 leakage-corrected
   dump. See `MANIFEST.json` in that folder for hashes vs the public repo.
2. `splits/` — thesis `model_train` / `valid` / `test` (seed 2026). Extra
   uses **union(train, valid)** for all statistics and neighbours.

Fingerprints: `../SHA256_DATA.json`.
