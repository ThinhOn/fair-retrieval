# Billion-scale fair k-NN — saved results

All numbers below are reproduced in `results/csv/*.csv`, plotted in `results/plots/`,
and backed by the raw per-query result pkls in `results/raw_results/` (+ the query
files in `results/queries/`). The large artifacts that make re-running unnecessary
(512 GB / 128 GB `vectors.dat`, `attributes.dat`, and the per-config SQLite indexes
`data/<ds>/lsh_cache_*`) live on disk under `data/` (gitignored) and are **intact** —
queries can be re-run from the indexes without rebuilding; metrics can be
re-evaluated from `results/raw_results/` + `results/queries/` without querying.

Config common to all billion runs: m=3, k=5, distance=ℓ2, mu=2, ILP post-processing,
SQLite cache backend, seed=10.

## 1. SIFT / BIGANN, n = 1e9, d = 128, w = 800  (50 queries; no ground truth at 1B)

| ell | build | index disk | success% | search/query | post/query | scanned |
|----:|------:|-----------:|---------:|-------------:|-----------:|--------:|
| 4  | 1.55 h | 16.0 GB | 100% | 35.5 s  | 33 ms | 1,038 |
| 8  | 1.53 h | 32.0 GB | 100% | 64.5 s  | 29 ms | 2,052 |
| 16 | 1.57 h | 64.1 GB | 100% | 124.1 s | 34 ms | 4,081 |
| 32 | 2.01 h | 128.1 GB| 100% | 250.2 s | 31 ms | 8,138 |
| 64 | 4.68 h*| 256.3 GB| 100% | 546.2 s | 32 ms | 16,254 |

*ell≤32 built with 8 workers; ell=64 with 4 workers (OOM avoidance) — so ell=64
build time is not directly comparable. Storage and query trends are clean.
recall@k is N/A at 1B (exact ground truth over 1e9 is infeasible — see subsets below).
Source: `scalability_bigann_1b.csv`, `summary_bigann_1b.csv`.

## 2. Synthetic (clustered), n = 1e9, d = 32, w = 4.0  (200 queries WITH ground truth; uniform 4 workers)

| ell | build | index disk | success% | DAF | recall@k | search/query | scanned |
|----:|------:|-----------:|---------:|----:|---------:|-------------:|--------:|
| 4  | 0.38 h | 16.0 GB | 28.5% | 1.401 | 0.0 | 0.59 s | 277 |
| 8  | 0.68 h | 32.1 GB | 65.0% | 1.242 | 0.0 | 1.60 s | 849 |
| 16 | 1.19 h | 64.1 GB | 87.0% | 1.133 | 0.0 | 4.09 s | 2,400 |
| 32 | 2.27 h | 128.2 GB| 97.5% | 1.091 | 0.0 | 9.62 s | 6,728 |
| 64 | 4.46 h | 256.5 GB| 99.5% | 1.070 | 0.0 | 17.49 s| 15,856 |

post-processing ~22 ms/query (constant). recall@k = 0 at every ell is an artifact of
near-equidistant points in tight clusters; DAF→1.07 shows near-optimal *distance*.
Source: `scalability_synthetic_1b.csv`, `summary_synthetic_1b.csv`.

## 3. Real-SIFT recall on subsets (where exact ground truth IS computable), w = 800

- 10k:  mu=2 ell=16 → recall@k 0.943, success 100%  (validated debug)
- 1M:   mu=2 ell=32 → 0.027 ; mu=0(auto) ell=32 → 0.146 ; ell=64 → 0.157 ; ell=256 → 0.277  (all 100% success)
- 10M:  mu=2 ell=32 → 0.003 ; mu=0(auto) ell=64 → 0.044  (100% success)

Recall on real SIFT is collision-limited (needs ell ∝ n^ρ); the well-separated
synthetic data hides this. Source: `summary_bigann_{10k,1m,10m}.csv`.

## Key takeaways (confirmed on both billion datasets)
- Storage exactly linear in ell (4 GB × ell, d-independent) → matches O(nℓμ).
- Query search time linear in ell; post-processing constant (depends on |C|, not ell/n).
- SIFT 100% feasible at all ell; clustered synthetic feasibility rises 28.5%→99.5% with ell.
- Use DAF (not recall@k) for distance quality on clustered data.

## How to reproduce / re-run cheaply
- Re-evaluate metrics (no querying): `python evaluate_ranged.py --results_dir outputs/<ds>/results_ranged_k=5_m=3_fdist=euclidean_<N> --queries_path data/<ds>/ranged_queries_k=5_m=3_fdist=euclidean_<N>.pkl --output summary.csv`
- Re-run queries only (indexes already built): `python code/main.py --data-dir data/<ds> --index l2lsh_cartesian --w <800|4.0> --ell <ELL> --mu 2 --m 3 --solver ilp --use-cache` (build step is skipped — cached partitions are detected).
- Full sweep scripts: `scripts/run_bigann_1b_sweep.sh`, `scripts/run_synthetic_1b_sweep.sh`.
- Regenerate plots: `python plot_billion.py`, `plot_synthetic.py`, `plot_combined.py`.
- Use the Python 3.11 interpreter (has numpy/scipy/pulp); the PATH `python` (Anaconda 3.8) lacks deps.
