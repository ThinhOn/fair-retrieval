import sys
sys.path.extend([
    'code/',
])
import json
import tqdm
import heapq
try:
    import torch
except ModuleNotFoundError:
    torch = None
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

# ── billion-scale schema (must match generate_billion.py) ──────────────────
ATTR_NAMES  = ["A1", "A2", "A3"]
ATTR_VALUES = [
    ["V11", "V12", "V13"],
    ["V21", "V22", "V23", "V24"],
    ["V31", "V32", "V33"],
]

seed = 10
set_seed(seed)
rng = default_rng(seed)
device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'


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
                   metadata_store: np.ndarray,
                   attrs_memmap=None,
                   attr_counts: dict = None,
                   partition_counts: dict = None,
                   attr_index: dict = None) -> tuple[bool, str]:
    """
    Fast structural feasibility check for ranged queries.

    Two paths:
    - Standard: uses df_meta (DataFrame) and metadata_store (string array).
    - Billion-scale: uses attr_counts {attr: {val: int}} and
      partition_counts {(code_tuple): int}, both pre-computed from attrs_memmap.
      metadata_store is ignored in this path.

    Three conditions must hold:
    1. Marginal availability: dataset has at least lb items for each mandatory (attr, val).
    2. Range-sum bounds: Σ lb_v ≤ k ≤ Σ ub_v per attribute.
    3. Intersectional availability: at least one item satisfies all mandatory tokens.
    """
    k = query['k']
    counts = query['count']     # { attr: { val: (lb, ub) } }

    # ── Check 1: marginal availability ───────────────────────────────────────
    for attr, val_ranges in counts.items():
        for val, (lb, ub) in val_ranges.items():
            if lb <= 0:
                continue
            if attr_counts is not None:
                # billion-scale: use pre-computed integer counts
                available = attr_counts.get(attr, {}).get(val, 0)
            elif attr_index is not None:
                # standard fast path: index lookup
                available = len(attr_index.get(f"{attr}:{val}", ()))
            else:
                # standard: use DataFrame frequency table
                if attr not in df_meta.columns:
                    return False, f"attribute '{attr}' not in dataset"
                freq = df_meta[attr].value_counts()
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
    mandatory_token_lists = [
        [f"{attr}:{val}" for val, (lb, ub) in val_ranges.items() if lb > 0]
        for attr, val_ranges in counts.items()
    ]
    mandatory_token_lists = [lst for lst in mandatory_token_lists if lst]

    for tpl in itt.product(*mandatory_token_lists):
        if partition_counts is not None:
            # billion-scale: look up the intersection count via integer code tuple
            code_key = []
            valid = True
            for token in sorted(tpl):   # sorted to match partition_counts key order
                attr, val = token.split(":", 1)
                j = ATTR_NAMES.index(attr)
                try:
                    code_key.append(ATTR_VALUES[j].index(val))
                except ValueError:
                    valid = False
                    break
            found = valid and partition_counts.get(tuple(code_key), 0) > 0
        elif attr_index is not None:
            # standard fast path: non-empty set intersection
            sets = [attr_index.get(tok) for tok in tpl]
            if not sets:
                found = True                      # no mandatory tokens
            elif any(not s for s in sets):
                found = False
            else:
                acc = set(sets[0])
                for s in sets[1:]:
                    acc &= s
                    if not acc:
                        break
                found = bool(acc)
        else:
            # standard: scan metadata strings
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


def _reconstruct_meta(oid: int, attrs_memmap) -> str:
    """Reconstruct metadata string from integer attribute codes (billion-scale)."""
    parts = [f"id:{oid}"]
    for j, code in enumerate(attrs_memmap[oid]):
        parts.append(f"{ATTR_NAMES[j]}:{ATTR_VALUES[j][int(code)]}")
    return "__".join(parts)


def _matching_ids_billion(tpl: tuple, attrs_memmap, chunk_size: int = 1_000_000):
    """
    Return ids whose integer attribute codes match all tokens in tpl.
    tpl: e.g. ('A1:V12', 'A2:V23')
    Uses integer comparison — O(n) but no string allocation.
    """
    # parse tokens into (attr_index, value_index) pairs
    filters = []
    for token in tpl:
        attr, val = token.split(":", 1)
        j = ATTR_NAMES.index(attr)
        v = ATTR_VALUES[j].index(val)
        filters.append((j, v))

    n = attrs_memmap.shape[0]
    matched = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        batch = attrs_memmap[start:end]          # (chunk, m) uint8
        mask = np.ones(end - start, dtype=bool)
        for j, v in filters:
            mask &= (batch[:, j] == v)
        local_ids = np.where(mask)[0]
        matched.append(local_ids + start)
    return np.concatenate(matched) if matched else np.array([], dtype=np.int64)


def build_attr_index(metadata_store):
    """One-pass index for the standard path: token 'attr:val' -> set(ids),
    plus id -> metadata string. Lets ground-truth/precheck use set intersections
    instead of rescanning all metadata strings per Cartesian tuple."""
    attr_index = collections.defaultdict(set)
    id2meta = {}
    for m in metadata_store:
        parts = str(m).split("__")
        oid = int(parts[0].split(":")[1])
        id2meta[oid] = m
        for tok in parts[1:]:
            attr_index[tok].add(oid)
    return attr_index, id2meta


def ground_truth_ilp(query, vector_store, metadata_store, dfunc, args,
                     attrs_memmap=None, attr_index=None, id2meta=None):
    """
    Compute the ground-truth fair k-NN for a ranged query using the ILP solver.

    Two paths:
    - Standard (attrs_memmap=None): scans metadata_store strings.
    - Billion-scale (attrs_memmap set): integer comparison over attrs_memmap,
      no string allocation during candidate search.

    query['count'] has the form { attr: { val: (lb, ub) } }.
    Returns (chosen, total_cost) or (None, None) if infeasible.
    """
    k = int(query['k'])

    # ── 1) Enumerate queried Cartesian tuples ────────────────────────────────
    attr_values = []
    for attr, val_ranges in query['count'].items():
        attr_values.append([f"{attr}:{val}" for val in val_ranges.keys()])
    carte_tuples = list(itt.product(*attr_values))

    # ── 2) For each tuple, collect matching ids and keep top-2k by distance ──
    per_tuple_topk = []
    for tpl in carte_tuples:
        if attrs_memmap is not None:
            # ── billion-scale: integer comparison ────────────────────────────
            match_ids = _matching_ids_billion(tpl, attrs_memmap)
            if len(match_ids) == 0:
                per_tuple_topk.append([])
                continue
            dists = [
                (int(oid), float(dfunc(query['vector'], vector_store[int(oid)])))
                for oid in match_ids
            ]
        elif attr_index is not None:
            # ── standard fast path: intersect id sets + vectorized distance ──
            idsets = [attr_index.get(tok) for tok in tpl]
            if any(not s for s in idsets):
                per_tuple_topk.append([])
                continue
            mids = set(idsets[0])
            for s in idsets[1:]:
                mids &= s
            if not mids:
                per_tuple_topk.append([])
                continue
            mids = np.fromiter(mids, dtype=np.int64)
            qv = np.asarray(query['vector'], dtype=np.float32)
            X = np.asarray(vector_store[mids], dtype=np.float32)
            if args.fdist == "cosine":
                Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
                dd = 1.0 - Xn @ (qv / (np.linalg.norm(qv) + 1e-12))
            else:
                # fdist "euclidean" => euclidean_dist_square (SQUARED L2), matching
                # get_dist_func used by the framework. NOT sqrt — must be the same
                # metric the search/objective use, or DAF is inconsistent.
                diff = X - qv
                dd = np.einsum("ij,ij->i", diff, diff)
            order = np.argsort(dd)[: k * 2]
            dists = [(int(mids[i]), float(dd[i])) for i in order]
            per_tuple_topk.append(dists)
            continue
        else:
            # ── standard: string matching (slow fallback) ─────────────────
            matches = [m for m in metadata_store if all(tok in m for tok in tpl)]
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

    # ── 3) Union: keep best (min) cost per id ────────────────────────────────
    id2best = {}
    for lst in per_tuple_topk:
        for oid, cost in lst:
            if id2best.get(oid) is None or cost < id2best[oid]:
                id2best[oid] = cost

    if not id2best:
        return None, None

    # ── 4) Build candidates with metadata strings ─────────────────────────────
    if attrs_memmap is not None:
        candidates = [
            (_reconstruct_meta(oid, attrs_memmap), cost)
            for oid, cost in id2best.items()
        ]
    else:
        if id2meta is None:
            id2meta = {int(m.split('__')[0].split(':')[1]): m for m in metadata_store}
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

    # ── detect billion-scale dataset ──────────────────────────────────────────
    import os
    vec_dat  = os.path.join(PATH, "vectors.dat")
    attr_dat = os.path.join(PATH, "attributes.dat")
    is_billion = os.path.exists(vec_dat) and os.path.exists(attr_dat)

    if is_billion:
        print("Detected billion-scale dataset (vectors.dat + attributes.dat)")
        meta_info    = np.load(os.path.join(PATH, "metadata.npz"), allow_pickle=True)
        n_total      = int(meta_info["n"][0])
        d_total      = int(meta_info["d"][0])
        m_attrs      = int(meta_info["attr_sizes"].shape[0])
        vector_store = np.memmap(vec_dat,  dtype=np.float32, mode="r",
                                 shape=(n_total, d_total))
        attrs_memmap = np.memmap(attr_dat, dtype=np.uint8,   mode="r",
                                 shape=(n_total, m_attrs))
        metadata_store = None
        df_meta        = None
        attr_index     = None
        id2meta        = None

        # attribute dict: {attr_name: [val_str, ...]}
        attributes_dict = {
            ATTR_NAMES[j]: ATTR_VALUES[j]
            for j in range(m_attrs)
        }

        # pre-compute marginal counts from attrs_memmap (O(n), one pass)
        print("Computing attribute value counts from attrs_memmap ...")
        attr_counts = {name: {val: 0 for val in vals}
                       for name, vals in attributes_dict.items()}
        partition_counts = {}   # {(code_tuple): count}
        chunk = 1_000_000
        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            batch = attrs_memmap[start:end]
            for local_i in range(end - start):
                codes = tuple(int(c) for c in batch[local_i])
                # marginal counts
                for j, name in enumerate(ATTR_NAMES[:m_attrs]):
                    attr_counts[name][ATTR_VALUES[j][codes[j]]] += 1
                # partition counts
                partition_counts[codes] = partition_counts.get(codes, 0) + 1
        print("Done computing counts.")

    else:
        # ── standard path ─────────────────────────────────────────────────────
        attrs_memmap     = None
        attr_counts      = None
        partition_counts = None
        attr_index       = None
        id2meta          = None

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

        # fast index for ground-truth (set intersections vs full metadata scans)
        print("Building attribute index for fast ground truth ...")
        attr_index, id2meta = build_attr_index(metadata_store)

    BATCH = 800
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
        ok, reason = precheck_query(
            q, df_meta, metadata_store,
            attrs_memmap=attrs_memmap,
            attr_counts=attr_counts,
            partition_counts=partition_counts,
            attr_index=attr_index,
        )
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
            query, vector_store, metadata_store, dfunc=dfunc, args=args,
            attrs_memmap=attrs_memmap, attr_index=attr_index, id2meta=id2meta,
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