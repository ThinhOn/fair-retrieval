import sys
sys.path.extend([
    'code/',
])
import json
import tqdm
import heapq
import torch
import pickle
import random
import argparse
import numpy as np
import collections
import pandas as pd
import itertools as itt
import multiprocessing as mp
from functools import partial
from numpy.random import default_rng

from utils import get_dist_func, set_seed, summarize_metadata
from solver import build_solver

seed = 10
set_seed(seed)
rng = default_rng(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_complex_queries(
    attributes_dict: dict[str, list[str]],
    num_examples: int,
    k: int,
    m: int,
    prob_options: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
) -> list[dict]:
    """
    Generate ranged fairness queries.

    Each attribute value v gets a count interval [lb_v, ub_v] instead of an
    exact count.  The construction guarantees Σ lb_v ≤ k ≤ Σ ub_v (feasibility
    precondition) by first drawing an exact centre count c_v with Σ c_v = k,
    then independently widening each interval by a random slack s_v:
        lb_v = max(0, c_v - s_v)
        ub_v = c_v + s_v
    Because lb_v ≤ c_v ≤ ub_v for every v, summing over v gives Σ lb_v ≤ k ≤ Σ ub_v.
    """
    attrs = list(attributes_dict.keys())
    num_attrs = len(attrs)

    assert m <= num_attrs, "m should be less than the number of attributes"
    carte_attrs = list(itt.combinations(range(num_attrs), m))

    queries = []
    for subset in carte_attrs:
        for _ in range(num_examples):
            query = {}
            query['k'] = k
            query['count'] = {}

            for attr_idx in subset:
                attr = attrs[attr_idx]
                attr_values = random.sample(attributes_dict[attr],
                                            k=min(3, len(attributes_dict[attr])))

                # ── Step 1: draw exact centre counts summing to k ──────────
                probs = [0]
                while sum(probs) != 1:
                    probs = random.choices(prob_options, k=len(attr_values))

                exact_counts = {
                    val: int(prob * k)
                    for val, prob in zip(attr_values, probs)
                }

                # Remove zeros and re-apportion remainder to a random value
                exact_counts = {v: c for v, c in exact_counts.items() if c > 0}
                if not exact_counts:
                    continue

                remainder = k - sum(exact_counts.values())
                if remainder != 0:
                    rand_val = random.choice(list(exact_counts.keys()))
                    exact_counts[rand_val] += remainder

                # ── Step 2: widen each centre count into [lb, ub] ─────────
                # slack s drawn uniformly from [0, c_v] so the interval is
                # always non-trivial but lb_v >= 0 is guaranteed.
                ranged = {}
                for val, c in exact_counts.items():
                    s = random.randint(0, max(1, c))   # at least width 1
                    lb = max(0, c - s)
                    ub = c + s
                    ranged[val] = (lb, ub)

                query['count'][attr] = ranged

            if query not in queries:
                queries.append(query)

    return queries


def precheck_query(query: dict, df_meta: pd.DataFrame,
                   metadata_store: np.ndarray) -> tuple[bool, str]:
    """
    Fast structural feasibility check for ranged queries.

    query['count'] has the form:
        { attr: { val: (lb, ub) } }

    Three conditions must hold:

    1. Marginal availability
       For every (attr, val) with lb > 0, the dataset must contain at least
       lb items with attr=val.

    2. Range-sum bounds
       For each attribute: Σ lb_v ≤ k ≤ Σ ub_v.

    3. Intersectional availability
       For every Cartesian tuple of (attr:val) tokens whose lb > 0, at least
       one dataset item must satisfy ALL tokens simultaneously.
    """
    k = query['k']
    counts = query['count']     # { attr: { val: (lb, ub) } }

    # ── Check 1: marginal availability ───────────────────────────────────────
    for attr, val_ranges in counts.items():
        if attr not in df_meta.columns:
            return False, f"attribute '{attr}' not in dataset"
        freq = df_meta[attr].value_counts()
        for val, (lb, ub) in val_ranges.items():
            if lb <= 0:
                continue
            # metadata values are stored as "attr:val" e.g. "A1:V12"
            raw_val = val.split(':')[-1]
            prefixed_val = f"{attr}:{raw_val}"
            available = int(freq.get(prefixed_val, freq.get(raw_val, freq.get(val, 0))))
            if available < lb:
                return False, (
                    f"not enough items: need at least {lb} × {attr}={val}, "
                    f"dataset has {available}"
                )

    # ── Check 2: range-sum bounds ─────────────────────────────────────────────
    for attr, val_ranges in counts.items():
        sum_lb = sum(lb for lb, ub in val_ranges.values())
        sum_ub = sum(ub for lb, ub in val_ranges.values())
        if not (sum_lb <= k <= sum_ub):
            return False, (
                f"infeasible range sums for '{attr}': "
                f"Σlb={sum_lb}, k={k}, Σub={sum_ub}; need Σlb ≤ k ≤ Σub"
            )

    # ── Check 3: intersectional availability (mandatory groups only) ──────────
    # Only consider values with lb > 0; optional values (lb=0) need not appear.
    mandatory_token_lists = [
        [f"{attr}:{val}" for val, (lb, ub) in val_ranges.items() if lb > 0]
        for attr, val_ranges in counts.items()
    ]
    # Skip attributes with no mandatory values
    mandatory_token_lists = [lst for lst in mandatory_token_lists if lst]

    for tpl in itt.product(*mandatory_token_lists):
        found = any(
            all(tok in meta for tok in tpl)
            for meta in metadata_store
        )
        if not found:
            return False, (
                f"no dataset item satisfies mandatory intersection {tpl}"
            )

    return True, "ok"


def parse_metadata(meta_str):
    """Convert metadata string into dict."""
    return dict(part.split(":") for part in meta_str.split("__"))


def satisfies(combination, query_counts):
    """
    Check if a combination satisfies ranged query counts.
    query_counts: { attr: { val: (lb, ub) } }
    """
    counts = {attr: collections.Counter() for attr in query_counts}
    for item in combination:
        meta = parse_metadata(item)
        for attr, needed in query_counts.items():
            if meta[attr] in needed:
                counts[attr][meta[attr]] += 1

    for attr, needed in query_counts.items():
        for value, (lb, ub) in needed.items():
            c = counts[attr][value]
            if not (lb <= c <= ub):
                return False
    return True


def ground_truth_ilp(query, vector_store, metadata_store, dfunc, args):
    """
    Compute the ground-truth fair k-NN for a ranged query using the ILP solver.

    query['count'] has the form { attr: { val: (lb, ub) } }.

    Returns (chosen, total_cost) or (None, None) if infeasible.
    """
    k = int(query['k'])

    # ── 1) Enumerate queried Cartesian tuples ────────────────────────────────
    attr_values = []
    for attr, val_ranges in query['count'].items():
        attr_values.append([f"{attr}:{val}" for val in val_ranges.keys()])
    carte_tuples = list(itt.product(*attr_values))

    # ── 2) Build id → metadata map ───────────────────────────────────────────
    id2meta = {int(m.split('__')[0].split(':')[1]): m for m in metadata_store}

    # ── 3) For each Cartesian tuple, collect matching ids and keep top-2k ────
    per_tuple_topk = []
    for tpl in carte_tuples:
        matches = [
            m for m in metadata_store
            if all(tok in m for tok in tpl)
        ]
        if not matches:
            per_tuple_topk.append([])
            continue

        dists = [
            (int(m.split('__')[0].split(':')[1]),
             float(dfunc(query['vector'], vector_store[int(m.split('__')[0].split(':')[1])])))
            for m in matches
        ]
        dists = heapq.nsmallest(k * 2, dists, key=lambda x: x[1])
        per_tuple_topk.append(dists)

    # ── 4) Union: keep best (min) cost per id ────────────────────────────────
    id2best = {}
    for lst in per_tuple_topk:
        for oid, cost in lst:
            if id2best.get(oid) is None or cost < id2best[oid]:
                id2best[oid] = cost

    if not id2best:
        return None, None

    candidates = [
        (id2meta[oid], cost)
        for oid, cost in id2best.items()
        if oid in id2meta
    ]

    # ── 5) Call ILP solver ────────────────────────────────────────────────────
    solver = build_solver(args)
    result = solver.solve(candidates, query)

    if not result or not result.get('selected'):
        return None, None

    # result['selected'] is a list of (meta_str, distance) tuples
    chosen = result['selected']
    chosen_ids = [int(m[0].split('__')[0].split(':')[1]) for m in chosen]
    total = sum(id2best[i] for i in chosen_ids if i in id2best)

    if total == 0:
        return None, None

    query['ground_truth'] = chosen
    query['ground_truth_dist'] = total
    return chosen, float(total)


def parser():
    parser = argparse.ArgumentParser(description="Creating queries")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fdist",   type=str, default="euclidean")
    parser.add_argument("--k",       type=int, default=10)
    parser.add_argument("--m",       type=int, default=3)
    parser.add_argument("--target",  type=int, default=200)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parser()
    DATASET = args.dataset
    PATH = f"./data/{DATASET}"

    npz_data = np.load(f'{PATH}/vectors.npz')
    vector_store, metadata_store = npz_data['vectors'], npz_data['metadata']

    if "celeb" in DATASET:
        print(f"load synthetic attributes with m={args.m}")
        metadata_store = np.load(f"{PATH}/metadata_m={args.m}.npz")['metadata']

    if "paper" in DATASET:
        vector_store = vector_store[:1000000]
        metadata_store = metadata_store[:1000000]

    df_meta = summarize_metadata(metadata_store)

    attributes_dict = {
        k: df_meta[k].apply(lambda x: x.split(':')[1]).unique().tolist()
        for k in df_meta.columns
    }

    BATCH = 80
    queries = []
    seen = set()

    while len(queries) < args.target:
        fresh = generate_complex_queries(
            attributes_dict,
            num_examples=BATCH,
            k=args.k,
            m=args.m,
            prob_options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
        )
        for q in fresh:
            # Stable signature using (lb, ub) tuples
            sig = (q['k'], tuple(sorted(
                (attr, tuple(sorted(
                    (val, lb, ub)
                    for val, (lb, ub) in v.items()
                )))
                for attr, v in q['count'].items()
            )))
            if sig not in seen:
                seen.add(sig)
                queries.append(q)
        print(f"Accumulated {len(queries)} unique queries...")

    dfunc = get_dist_func(args.fdist)

    for query in tqdm.tqdm(queries):
        idx = int(rng.integers(0, len(vector_store)))
        query['vector'] = vector_store[idx]

    # ── Pre-filter ────────────────────────────────────────────────────────────
    prechecked = []
    n_rejected = 0
    for q in queries:
        ok, reason = precheck_query(q, df_meta, metadata_store)
        if ok:
            prechecked.append(q)
        else:
            n_rejected += 1
    print(f"[Precheck] {len(prechecked)} passed / {n_rejected} rejected")
    queries = prechecked

    args.solver = "ilp"
    count = 0
    valid_query_idx = []
    for idx, query in enumerate(tqdm.tqdm(queries, desc="Computing ground truth", unit="query")):
        result = ground_truth_ilp(
            query, vector_store, metadata_store, dfunc=dfunc, args=args
        )
        if result is None or result == (None, None):
            continue
        if 'ground_truth' in query:
            count += 1
            valid_query_idx.append(idx)
            if count >= args.target:
                break

    queries = [queries[idx] for idx in valid_query_idx]

    print(f"Total final queries: {len(queries)}")
    # if queries:
    #     print(queries[0])

    with open(
        f'{PATH}/ranged_queries_k={args.k}_m={args.m}_fdist={args.fdist}_{len(queries)}.pkl',
        'wb'
    ) as f:
        pickle.dump(queries, f)
