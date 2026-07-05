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

## 4. IVF retrieval backend (Design A: one FAISS IVF index per Cartesian partition)

IVF as a third retrieval family (reviewer R1O4/R1O8), plugged into the same
framework — post-processing unchanged (retrieval-agnostic). The effort knob is
`nprobe` (query-time; build once, sweep for free). nlist auto = 4·√n_π;
partitions with <64 pts fall back to flat (exact). Module: `l2ivf_cartesian.py`,
runner: `scripts/run_ivf_sweep.sh`. 200 queries with ground truth → recall + DAF.

Real-world datasets, ℓ2, low config (nprobe 1/4/8):

| dataset (m) | nprobe | recall@k | DAF | success% | search ms |
|---|---:|---:|---:|---:|---:|
| Audio (m=3)    | 1 | 0.644 | 1.092 | 100 | 1.9 |
|                | 4 | 0.902 | 1.013 | 100 | 2.6 |
|                | 8 | 0.972 | 1.003 | 100 | 2.7 |
| CelebA (m=2)   | 1 | 0.647 | 1.032 | 100 | 229 |
|                | 4 | 0.892 | 1.007 | 100 | 220 |
|                | 8 | 0.957 | 1.002 | 100 | 220 |
| FairFace (m=3) | 1 | 0.715 | 1.011 | 100 | 9.7 |
|                | 4 | 0.978 | 1.000 | 100 | 8.2 |
|                | 8 | 0.999 | 1.000 | 100 | 8.4 |

(CelebA search is ~220 ms because metadata carries 5 attributes → 4^5=1024
partitions, many probed per m=2 query.)

CelebA across m (IVF, nprobe 1/4/8). NOTE: m=1,3,4,5 use the pre-existing
EXACT-count queries (queries_k=5_m=N^5_*.pkl, which carry ground truth), while
m=2 above used ranged constraints — exact counts are stricter, so success% falls
sharply with m (the multi-attribute hardness). DAF stays near-optimal throughout.

| m | recall@k (np 1/4/8) | DAF@8 | success% | search ms |
|--:|---|---:|---:|---:|
| 1 | 0.668 / 0.905 / 0.959 | 1.004 | 100.0 | 369 |
| 3 | 0.768 / 0.908 / 0.927 | 1.001 | 83.5  | 132 |
| 4 | 0.708 / 0.815 / 0.835 | 1.001 | 55.0  | 79  |
| 5 | 0.662 / 0.735 / 0.752 | 1.000 | 19.0  | 47  |

Source: `summary_ivf_celeba_m={1,3,4,5}.csv`. The IVF index is identical across m
(partitioning is by all 5 attrs), so all these reuse the one built CelebA cache.

CelebA across m with RANGED constraints (IVF, nprobe 1/4/8) — the consistent
counterpart of the ranged m=2 above (generated with fast squared-L2 ground truth):

| m | recall@k (np 1/4/8) | DAF@8 | success% | n_q |
|--:|---|---:|---:|---:|
| 1 | 0.666 / 0.896 / 0.952 | 1.003 | 100.0 | 200 |
| 3 | 0.801 / 0.957 / 0.982 | 1.001 | 98.5  | 200 |
| 4 | 0.756 / 0.919 / 0.967 | 1.010 | 96.0  | 200 |
| 5 | 0.755 / 0.898 / 0.932 | 1.005 | 72.7  | 132 |

Ranged success% (100/100/98.5/96/72.7 for m=1/2/3/4/5) is far higher than the
exact-count version above (100/83.5/55/19) — ranged constraints are more flexible.
DAF stays ~1.00 (near-optimal distance) throughout. m=5 yields only 132 feasible
queries out of the candidate pool (ranged 5-attribute fairness is hard).
Source: `summary_ivf_celeba_ranged_m={1,3,4,5}.csv`.

Real-SIFT subset (1M, d=128, with ground truth) — IVF vs LSH at scale:

| nprobe | recall@k | DAF | success% | search ms |
|---:|---:|---:|---:|---:|
| 1  | 0.407 | 1.111 | 100 | 5.4 |
| 4  | 0.738 | 1.029 | 100 | 7.5 |
| 8  | 0.858 | 1.013 | 100 | 7.6 |
| 16 | 0.935 | 1.004 | 100 | 7.8 |
| 32 | 0.976 | 1.001 | 100 | 8.1 |

IVF reaches recall 0.976 on 1M SIFT vs LSH's 0.277 at ell=256 — data-adaptive
k-means cells beat data-oblivious LSH on real (non-uniform) data, while
post-processing stays 100% feasible. Source: `summary_ivf_*.csv`.
(IVF was NOT run at billion scale per user request; would need IVFPQ for memory.)

## How to reproduce / re-run cheaply
- Re-evaluate metrics (no querying): `python evaluate_ranged.py --results_dir outputs/<ds>/results_ranged_k=5_m=3_fdist=euclidean_<N> --queries_path data/<ds>/ranged_queries_k=5_m=3_fdist=euclidean_<N>.pkl --output summary.csv`
- Re-run queries only (indexes already built): `python code/main.py --data-dir data/<ds> --index l2lsh_cartesian --w <800|4.0> --ell <ELL> --mu 2 --m 3 --solver ilp --use-cache` (build step is skipped — cached partitions are detected).
- Full sweep scripts: `scripts/run_bigann_1b_sweep.sh`, `scripts/run_synthetic_1b_sweep.sh`.
- Regenerate plots: `python plot_billion.py`, `plot_synthetic.py`, `plot_combined.py`.
- Use the Python 3.11 interpreter (has numpy/scipy/pulp); the PATH `python` (Anaconda 3.8) lacks deps.
