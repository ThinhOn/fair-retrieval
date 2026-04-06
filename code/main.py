import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import tqdm
import math
import time
import json
import pickle
import pandas as pd
import numpy as np

sys.path.extend([
    '.',
    'code/indexing/',
])

from indexing import build_index
from arguments import get_args
from utils import (
    set_seed,
    get_dist_func,
)


## TODO: make sure to vary hashing parameters when working with billion scale data
## and measure run time, memory cost.
## query processing: cost time, quality measures.
## TODO: anonymous GITHUB
## TODO: more related work recently (accepted tO VLDB, SIGMOD, ICDE 26, filtering out vecDB papers)
## TODO: create SIGMOD submission

if __name__ == "__main__":

    args = get_args()
    set_seed(args.seed)
    
    DATASET = args.data_dir.split("/")[-1]

    # ── billion-scale path: vectors.dat + attributes.dat memmaps ─────────────
    vec_dat  = os.path.join(args.data_dir, "vectors.dat")
    attr_dat = os.path.join(args.data_dir, "attributes.dat")
    is_billion = os.path.exists(vec_dat) and os.path.exists(attr_dat)

    if is_billion:
        meta_info   = np.load(os.path.join(args.data_dir, "metadata.npz"), allow_pickle=True)
        n_total     = int(meta_info["n"][0])
        d_total     = int(meta_info["d"][0])
        vector_store  = np.memmap(vec_dat,  dtype=np.float32, mode="r", shape=(n_total, d_total))
        attrs_memmap  = np.memmap(attr_dat, dtype=np.uint8,   mode="r", shape=(n_total, len(meta_info["attr_sizes"])))
        metadata_store = None   # not used; reconstructed on the fly
        # protected_attrs built from ATTR_NAMES / ATTR_VALUES in l2lsh_cartesian
        from l2lsh_cartesian import ATTR_NAMES, ATTR_VALUES
        protected_attrs = []
        for j, name in enumerate(ATTR_NAMES):
            for val in ATTR_VALUES[j]:
                protected_attrs.append(f"{name}:{val}")
        print(f"Billion-scale dataset: {n_total:,} vectors, d={d_total}, attrs={attrs_memmap.shape[1]}")
    else:
        # ── standard path ─────────────────────────────────────────────────────
        attrs_memmap = None
        npz_data = np.load(f"{args.data_dir}/vectors.npz")
        vector_store, metadata_store = npz_data['vectors'], npz_data['metadata']

        if "celeb" in DATASET:
            metadata_store = np.load(f"{args.data_dir}/metadata_m={args.m}.npz")['metadata']

        if "paper" in DATASET:
            vector_store = vector_store[:1_000_000]
            metadata_store = metadata_store[:1_000_000]


    # """ uncomment this block for DEBUG
    # n_debug = 1000000
    # print(f"[Debug] Using first {n_debug:,} vectors")
    # vector_store  = vector_store[:n_debug]
    # if is_billion:
    #     attrs_memmap = attrs_memmap[:n_debug]
    #     n_total = n_debug
    # else:
    #     metadata_store = metadata_store[:n_debug]
    # """


    if args.fdist == "cosine":
        norms = np.linalg.norm(vector_store, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        vector_store = vector_store / norms

    if not is_billion:
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
    import psutil, os as _os
    proc = psutil.Process(_os.getpid())

    start = time.time()
    index = build_index(args)
    db = index(
        args,
        vector_store.shape[-1],
        metadata_store,
        protected_attrs,
        attrs_memmap=attrs_memmap,
    )
    mem_before = proc.memory_info().rss
    db.build_index(vector_store)
    indexing_time = time.time() - start

    if is_billion:
        size_mb = (proc.memory_info().rss - mem_before) / (1024 * 1024)
        print(f"Memory cost (RSS delta): {size_mb:.1f} (MB)")
    else:
        data = pickle.dumps(db, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = len(data) / (1024 * 1024)
        print(f"Memory cost: {size_mb} (MB)")
    print(f"Preprocessing time: {indexing_time} (s)")
    
    """
    retrieval
    """
    dfunc = get_dist_func(args.fdist)

    avg_post = 0.
    for k in [5,]:
    # for k in [5, 10, 15, 20, 25]:
        ## load query file
        # query_suffix = f"k={k}_m={args.m}^5_fdist={args.fdist}_200"
        query_suffix = f"k={k}_m={args.m}_fdist={args.fdist}_200"
        query_path = f"{args.data_dir}/ranged_queries_{query_suffix}.pkl"
        with open(query_path, 'rb') as f:
            queries = pickle.load(f)

        result_path = f"./outputs/{DATASET}/results_ranged_{query_suffix}/{INDEX}.pkl"
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))

        results = {}
        results["preprocessing_time"] = indexing_time
        results["index_memory_MB"] = size_mb

        results["query_results"] = []
        n_failed = 0
        for query in tqdm.tqdm(queries):
            result = db.search_and_solve(query, vector_store, dfunc)
            results["query_results"].append(result)
            if result is None:
                n_failed += 1
                continue
            avg_post += result["postprocessing_time"]

        if n_failed:
            print(f"[Warning] {n_failed}/{len(queries)} queries returned no candidates")

        with open(result_path, 'wb') as f:
            pickle.dump(results, f)

        # print(f"Post processing time for C_MIN={C_MIN}:")
        # print(avg_post/10)
        # print()