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
# from models.blip import blip_feature_extractor
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
    attrs = list(attributes_dict.keys())
    num_attrs = len(attrs)

    carte_attrs = []   # groups for producing cartesian products 
    # for r in range(2, 3):
    assert m <= num_attrs, "m should be less than the number of attributes"
    carte_attrs.extend(list(itt.combinations(range(num_attrs), m)))
    # print(carte_attrs)
    # exit()

    queries = []
    for subset in carte_attrs:
        # remaining_attrs = list(set(attrs) - set([attrs[ii] for ii in subset]))
        for _ in range(num_examples):
            query = {}
            """
            step 1: create a list of values for K and prob
            step 2: then randomly select
            """
            query['k'] = k
            query['count'] = {}
            for attr_idx in subset:
                attr = attrs[attr_idx]
                attr_values = attributes_dict[attr]
                attr_values = random.sample(attributes_dict[attr], k=min(3, len(attr_values)))
                probs = [0]
                while sum(probs) != 1:
                    probs = random.choices(prob_options, k=len(attr_values))
                query['count'][attr] = {
                    attr_val: int(prob * query['k'])
                    for attr_val, prob in zip(attr_values, probs)
                }
                ## delete attribute values with count = 0
                delete_attr_val = [attr_val for attr_val, count in query['count'][attr].items() if count == 0]
                for attr_val in delete_attr_val:
                    del query['count'][attr][attr_val]
            # print(query, '\n')
            for attr, sub_dict in query['count'].items():
                if sum(sub_dict.values()) < query['k']:
                    # then select a random attribute_value and add up so that the sum == k
                    remaining = query['k'] - sum(sub_dict.values())
                    rand_attr_val = random.choice(list(sub_dict.keys()))
                    query['count'][attr][rand_attr_val] = query['count'][attr].get(rand_attr_val, 0) + remaining
            if query not in queries:
                queries.append(query)
            # print(query)
            # exit()
    return queries



def validate_query(query, df_meta):
    """
    check if query is valid
    """
    groups = [k.split(':')[0] for k in next(iter(query['target_count']))]
    stats = df_meta.groupby(groups).groups
    stats = {tuple(sorted(k)): len(v) for k, v in stats.items()}
    try:
        return all([query['target_count'][k] <= stats[k] for k in query['target_count']])
    except KeyError:
        return False


def parse_metadata(meta_str):
    """Convert metadata string into dict"""
    return dict(part.split(":") for part in meta_str.split("__"))

def satisfies(combination, query_counts):
    """Check if a combination satisfies query counts"""
    counts = {attr: collections.Counter() for attr in query_counts}
    for item in combination:
        meta = parse_metadata(item)
        for attr, needed in query_counts.items():
            if meta[attr] in needed:
                counts[attr][meta[attr]] += 1

    # check constraints
    for attr, needed in query_counts.items():
        for value, required_count in needed.items():
            if counts[attr][value] < required_count:
                return False
    return True


def ground_truth_ilp(query, vector_store, metadata_store, dfunc, args):
    """
    For each queried Cartesian attribute tuple, select the top-k nearest items (by cost),
    take the union across tuples, then run the existing solver (ILP when m>=3).
    Returns (feasible: bool, chosen_ids: list|None, total_cost: float|None).
    """

    k = int(query['k'])

    # --- 1) Enumerate queried Cartesian tuples (intersectional groups) ---
    # query['count'] looks like: {'gender': {'male': 2, 'female': 3}, 'race': {...}, ...}
    attr_values = []
    for A, counts in query['count'].items():
        vals = [f"{A}:{v}" for v in counts.keys()]
        attr_values.append(vals)
    carte_tuples = list(itt.product(*attr_values))  # e.g., [('gender:male','race:Asian'), ('gender:female','race:Asian'), ...]

    # --- 2) Build id->metadata map once ---
    id2meta = {int(m.split('__')[0].split(':')[1]): m for m in metadata_store}

    # --- 3) For each tuple, collect ALL matching ids, score them, keep top-k by cost ---
    per_tuple_topk = []
    for tpl in carte_tuples:
        # match if ALL tokens in tpl are present in metadata string
        matches = []
        for m in metadata_store:
            if all(tok in m for tok in tpl):
                oid = int(m.split('__')[0].split(':')[1])
                matches.append(oid)

        if not matches:
            per_tuple_topk.append([])  # no candidates for this tuple
            continue

        # distance (cost) once
        dists = [(oid, float(dfunc(query['vector'], vector_store[oid]))) for oid in matches]

        # pick top-k by cost (if fewer than k exist, keep them all)
        if len(dists) > k:
            # use nsmallest by cost
            dists = heapq.nsmallest(k * 2, dists, key=lambda x: x[1])
        else:
            dists.sort(key=lambda x: x[1])

        per_tuple_topk.append(dists)

    # --- 4) Union by id, keep the best (min) cost per id to avoid duplicates ---
    id2best = {}
    for lst in per_tuple_topk:
        for oid, cost in lst:
            prev = id2best.get(oid, None)
            if prev is None or cost < prev:
                id2best[oid] = cost

    # If nothing to solve over, it's infeasible for this query
    if not id2best:
        return None, None

    # --- 5) Prepare candidates in your solver’s expected format: (metadata, cost) ---
    # (Your search path hands candidates to the solver exactly like this.) :contentReference[oaicite:2]{index=2}
    candidates = [(str(id2meta[oid]), cost) for oid, cost in id2best.items() if oid in id2meta]

    # --- 6) Call your existing solver (min-cost flow for m=2; ILP for m>=3) ---
    solver = build_solver(args)
    result = solver.solve(candidates, query)

    # if not result or not result.get('feasible', True):
    #     return None, None

    # Normalize output to ids and compute total cost
    chosen = result['selected']
    chosen_ids = [int(m[0].split('__')[0].split(':')[1]) for m in chosen]
    total = sum(id2best[i] for i in chosen_ids if i in id2best)

    if total == 0:
        query = None
        return 
    query['ground_truth'] = chosen
    query['ground_truth_dist'] = total
    return chosen, float(total)


def parser():
    parser = argparse.ArgumentParser(description="Creating queries")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset to use"
    )
    parser.add_argument(
        "--fdist", type=str, default="euclidean", help="Distance function"
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Query size k"
    )
    parser.add_argument(
        "--m", type=int, default=3, help="Number of protected attributes"
    )
    parser.add_argument(
        "--target", type=int, default=200, help="Number of queries to generate"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parser()

    DATASET = args.dataset

    PATH = f"./data/{DATASET}"

    npz_data = np.load(f'{PATH}/vectors.npz')
    vector_store, metadata_store = npz_data['vectors'], npz_data['metadata']

    if "celeb" in DATASET:
        print("load synthetic attributes with m=5")
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
            prob_options=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
        )
        # Deduplicate by a stable signature (counts + k)
        for q in fresh:
            sig = (q['k'], tuple(sorted(
                (attr, tuple(sorted(v.items())))
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

    args.solver = "ilp"
    count = 0
    valid_query_idx = []
    for idx, query in enumerate(tqdm.tqdm(queries, desc="Computing ground truth", unit="query")):
        ground_truth_ilp(
            query,
            vector_store,
            metadata_store,
            dfunc=dfunc,
            args=args
        )
        if query is None: continue
        if 'ground_truth' in query:
            count += 1
            valid_query_idx.append(idx)
            if count >= args.target: break
        
    # queries = [q for q in queries if q is not None and 'ground_truth' in q]
    # queries = queries[:args.target]
    queries = [queries[idx] for idx in valid_query_idx]

    with open(f'{PATH}/queries_k={args.k}_m={args.m}_fdist={args.fdist}_{len(queries)}.pkl', 'wb') as f:
        pickle.dump(queries, f)