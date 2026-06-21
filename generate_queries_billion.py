"""
generate_queries_billion.py
----------------------------
Lightweight ranged-query generator for the FULL 1B SIFT/BIGANN run.

Unlike generate_queries_ranged.py, this does NOT compute exact ground truth
(the ground-truth ILP scans all 1e9 vectors per Cartesian tuple — days of
compute) and does NOT run the O(n) Python precheck. At billion scale every
attribute value has hundreds of millions of points, so the only feasibility
condition that can bind — Σlb ≤ k ≤ Σub — is already guaranteed by the
construction in generate_complex_queries(). Marginal/intersectional
availability is satisfied trivially.

Output: data/<ds>/ranged_queries_k=<k>_m=<m>_fdist=<fdist>_<N>.pkl
(same filename main.py expects), with each query carrying 'k', 'count', and a
'vector' sampled from the dataset — but NO 'ground_truth' key. Evaluation at
1B therefore reports success%/timing/scanned (recall@k needs ground truth and
is reported on the 1M/10M subsets instead).
"""

import os
import sys
import pickle
import argparse
import numpy as np

sys.path.extend(['code/'])
from generate_queries_ranged import (
    generate_complex_queries, ATTR_NAMES, ATTR_VALUES,
)
from utils import set_seed

set_seed(10)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--fdist",   type=str, default="euclidean")
    p.add_argument("--k",       type=int, default=5)
    p.add_argument("--m",       type=int, default=3)
    p.add_argument("--target",  type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parser()
    PATH = f"./data/{args.dataset}"

    meta = np.load(os.path.join(PATH, "metadata.npz"), allow_pickle=True)
    n = int(meta["n"][0]); d = int(meta["d"][0])
    m_attrs = int(meta["attr_sizes"].shape[0])
    vectors = np.memmap(os.path.join(PATH, "vectors.dat"),
                        dtype=np.float32, mode="r", shape=(n, d))
    print(f"Dataset: n={n:,} d={d} m_attrs={m_attrs}")

    attributes_dict = {ATTR_NAMES[j]: ATTR_VALUES[j] for j in range(m_attrs)}

    # Generate unique feasible queries (Σlb ≤ k ≤ Σub guaranteed by construction)
    queries, seen = [], set()
    rng = np.random.default_rng(10)
    while len(queries) < args.target:
        fresh = generate_complex_queries(
            attributes_dict, num_examples=800, k=args.k, m=args.m,
            prob_options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
        )
        for q in fresh:
            sig = (q['k'], tuple(sorted(
                (a, tuple(sorted((v, lb, ub) for v, (lb, ub) in vm.items())))
                for a, vm in q['count'].items())))
            if sig not in seen:
                seen.add(sig); queries.append(q)
        print(f"Accumulated {len(queries)} unique queries...")
    queries = queries[:args.target]

    # Attach a query vector sampled from the dataset (one memmap read each)
    for q in queries:
        idx = int(rng.integers(0, n))
        q['vector'] = np.array(vectors[idx])

    out = f"{PATH}/ranged_queries_k={args.k}_m={args.m}_fdist={args.fdist}_{len(queries)}.pkl"
    with open(out, "wb") as f:
        pickle.dump(queries, f)
    print(f"Saved {len(queries)} queries (no ground truth) to {out}")
