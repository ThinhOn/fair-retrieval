import os
import sys
import tqdm
import math
import time
import json
import pickle
import pandas as pd
import numpy as np

sys.path.extend([
    'code/indexing/',
])

from indexing import build_index
from arguments import get_args
from utils import (
    set_seed,
    get_dist_func,
)


if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    
    DATASET = args.data_dir.split("/")[-1]

    npz_data = np.load(f"{args.data_dir}/vectors.npz")
    vector_store, metadata_store = npz_data['vectors'], npz_data['metadata']

    if "celeb" in DATASET:
        metadata_store = np.load(f"{args.data_dir}/metadata_m=5.npz")['metadata']

    if args.fdist == "cosine":
        norms = np.linalg.norm(vector_store, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        vector_store = vector_store / norms

    metadata_store = [str(md) for md in metadata_store]
    protected_attrs = [md.split('__')[1:] for md in metadata_store]
    protected_attrs = list(set( [attr for sublist in protected_attrs for attr in sublist] ))

    """
    Compute R
    """
    if "lsh" in args.index:
        # sample K points
        sample = vector_store[np.random.choice(len(vector_store), size=5000, replace=False)]
        nn_dists = []
        for i in range(len(sample)):
            d = np.linalg.norm(sample - sample[i], axis=1)
            d[i] = np.inf
            nn_dists.append(np.min(d))
        args.r = round(np.percentile(nn_dists, 50) * 2, 5)
        
    ## indexing params
    if "lsh" in args.index:
        INDEX = f"{args.index}/c={args.c}_r={args.r:.3f}_w={args.w}_ell={args.ell}_mu={args.mu}_delta={args.delta}"
    elif "sieve" in args.index:
        INDEX = f"{args.index}/efc={args.ef_construction}_M={args.M}_efs={args.ef_search}"
    elif "disk" in args.index:
        INDEX = f"{args.index}/efc={args.ef_construction}_M={args.M}_efs={args.ef_search}_mult={args.filtering_multiplier}"
    elif "brute" in args.index:
        INDEX = f"{args.index}"
    else:
        raise ValueError(f"Index {args.index} is not configured to save outputs! Check main.py!!")

    """
    indexing
    """
    start = time.time()
    index = build_index(args)
    db = index(
        args,
        vector_store[0].shape[-1],
        metadata_store,
        protected_attrs,
    )
    db.build_index(vector_store)

    data = pickle.dumps(db, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = len(data) / (1024 * 1024)
    print(f"Memory cost: {size_mb} (MB)")

    indexing_time = time.time() - start
    print(f"Preprocessing time: {indexing_time} (s)")
    
    """
    retrieval
    """
    dfunc = get_dist_func(args.fdist)

    for k in [10]:
    # for k in [5, 10, 15, 20]:
        ## load query file
        # query_suffix = f"k={k}_m={args.m}^5_fdist={args.fdist}_200"
        query_suffix = f"k={k}_m={args.m}^5_fdist={args.fdist}_200"
        query_path = f"{args.data_dir}/queries_{query_suffix}.pkl"
        with open(query_path, 'rb') as f:
            queries = pickle.load(f)

        result_path = f"./outputs/{DATASET}/results_{query_suffix}/{INDEX}.pkl"
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))

        results = {}
        results["preprocessing_time"] = indexing_time
        results["index_memory_MB"] = size_mb

        results["query_results"] = []
        count_nan = 0
        for query in tqdm.tqdm(queries):
            result = db.search_and_solve(query, vector_store, dfunc)
            if result is None or result.get('status', '') == "Infeasible":
                count_nan += 1
            results["query_results"].append(result)

        print(f"Total queries: {len(queries)}, Failed queries: {count_nan}")

        # for q, r in zip(queries[:5], results['query_results'][:5]):
        #     print("Query:", q['count'], q['ground_truth'])
        #     print("Result:", r)
        #     print('\n\n')
        # exit()
        
        # with open(result_path, 'wb') as f:
        #     pickle.dump(results, f)