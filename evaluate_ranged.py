"""
evaluate_ranged.py
------------------
Parse result pkl files and compute recall@k and percentage of successfully
solved queries for ranged fairness constraints.

Directory layout assumed:
    results_ranged_k={k}_m={m}_fdist={fdist}_{n}/
        {method}/
            {param_string}.pkl     ← list of per-query result dicts

Query files assumed at:
    data/{dataset}/ranged_queries_k={k}_m={m}_fdist={fdist}_{n}.pkl

Usage:
    python evaluate_ranged.py \
        --results_dir ./results_ranged_k=10_m=2_fdist=euclidean_200 \
        --queries_path ./data/celeba/ranged_queries_k=10_m=2_fdist=euclidean_200.pkl \
        --output summary.csv
"""

import os
import pickle
import argparse
import pandas as pd
from collections import defaultdict


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_id(meta_str):
    # result["selected"] may contain (meta_str, dist) tuples or plain strings
    if isinstance(meta_str, tuple):
        meta_str = meta_str[0]
    return int(meta_str.split("__")[0].split(":")[1])


def ground_truth_ids(query: dict) -> set[int]:
    """
    query['ground_truth'] is a list of (meta_str, dist) tuples.
    Returns the set of integer ids.
    """
    return {extract_id(m) for m, _ in query["ground_truth"]}


def selected_ids(result: dict) -> set[int]:
    """result['selected'] is a list of meta strings."""
    return {extract_id(m) for m in result["selected"]}


def is_feasible(result: dict, query: dict) -> bool:
    """
    Check whether result['count'] satisfies all ranged constraints in query['count'].
    query['count']  : {attr: {val: (lb, ub)}}
    result['count'] : {attr: {val: int}}
    """
    for attr, val_ranges in query["count"].items():
        result_counts = result.get("count", {}).get(attr, {})
        for val, (lb, ub) in val_ranges.items():
            got = result_counts.get(val, 0)
            if not (lb <= got <= ub):
                return False
    return True


def recall_at_k(result: dict, query: dict) -> float:
    """Fraction of ground-truth ids recovered in the result."""
    k = int(query["k"])
    if not result.get("selected"):
        return 0.0
    gt = ground_truth_ids(query)
    sel = selected_ids(result)
    return len(gt & sel) / k


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate_file(result_path: str, queries: list[dict]) -> dict:
    """
    Load one result pkl and compute per-query metrics.
    Returns a dict of aggregate metrics.
    """
    with open(result_path, "rb") as f:
        results = pickle.load(f)

    # Normalise to a list — three possible formats:
    #   (a) list of per-query dicts                          → use as-is
    #   (b) wrapper dict with 'query_results' key            → unwrap
    #   (c) dict keyed by integer query index (str or int)   → sort and extract values
    if isinstance(results, dict):
        if "query_results" in results:
            results = results["query_results"]
        else:
            results = [results[k] for k in sorted(results.keys(), key=lambda x: int(x))]

    # Guard: result list may be shorter than queries if some queries timed out
    n = min(len(results), len(queries))

    recalls = []
    feasible_flags = []
    search_times = []
    post_times = []
    scanned = []

    for i in range(n):
        res = results[i]
        q   = queries[i]

        # Treat empty / None result as failure
        if not res or not res.get("selected"):
            recalls.append(0.0)
            feasible_flags.append(False)
            continue

        recalls.append(recall_at_k(res, q))
        feasible_flags.append(is_feasible(res, q))
        search_times.append(res.get("search_time", 0.0))
        post_times.append(res.get("postprocessing_time", 0.0))
        scanned.append(res.get("total_scanned", 0))

    total = n
    success = sum(feasible_flags)

    return {
        "n_queries":        total,
        "recall@k":         sum(recalls) / total if total else 0.0,
        "success_%":        100.0 * success / total if total else 0.0,
        "avg_search_ms":    1000 * sum(search_times) / len(search_times) if search_times else 0.0,
        "avg_post_ms":      1000 * sum(post_times)   / len(post_times)   if post_times   else 0.0,
        "avg_scanned":      sum(scanned) / len(scanned) if scanned else 0.0,
    }


def parse_results_dir(results_dir: str, queries: list[dict]) -> pd.DataFrame:
    """
    Walk results_dir, evaluate every pkl, return a DataFrame with one row per file.
    """
    rows = []
    for method in sorted(os.listdir(results_dir)):
        method_dir = os.path.join(results_dir, method)
        if not os.path.isdir(method_dir):
            continue
        for fname in sorted(os.listdir(method_dir)):
            if not fname.endswith(".pkl"):
                continue
            fpath = os.path.join(method_dir, fname)
            params = fname.replace(".pkl", "")
            try:
                metrics = evaluate_file(fpath, queries)
            except Exception as e:
                import traceback
                print(f"[WARN] could not evaluate {fpath}:")
                traceback.print_exc()
                continue
            rows.append({
                "method":  method,
                "params":  params,
                **metrics,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["method", "params"]).reset_index(drop=True)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",  required=True,
                        help="Top-level result folder, e.g. results_ranged_k=10_m=2_fdist=euclidean_200")
    parser.add_argument("--queries_path", required=True,
                        help="Path to the matching ranged queries pkl")
    parser.add_argument("--output",       default="summary.csv",
                        help="Where to write the CSV summary (default: summary.csv)")
    args = parser.parse_args()

    print(f"Loading queries from {args.queries_path} ...")
    with open(args.queries_path, "rb") as f:
        queries = pickle.load(f)
    print(f"  {len(queries)} queries loaded.")

    print(f"Evaluating results in {args.results_dir} ...")
    df = parse_results_dir(args.results_dir, queries)

    print("\n" + df.to_string(index=False))
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
