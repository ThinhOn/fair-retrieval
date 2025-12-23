"""
Cartesian-Subindex ANN

Design goals
- Pure Python + NumPy fallback (BruteForce backend). Optional hnswlib hook.
- Clear interfaces so you can swap the ANN backend without touching orchestration.
- No reliance on ef_h or S(I_h); we use fixed k_sub per subindex, as requested.

Usage:
- Build with small toy data
- Query gender-only, race-only, or full Cartesian (gender ∧ race)

Notes
- Distance metric: cosine (default) or L2.
- If hnswlib is available, you can switch backend to "hnsw" to accelerate.
- Deduplicates IDs when merging results from multiple subindexes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Any, Set
import math
import time
import numpy as np
import itertools as itt
from collections import defaultdict
import hnswlib
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from solver import build_solver


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    return x


def parse_metadata_store(metadata_store: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    metadata_store element format:
      "id:<int>__attr1:<val>__attr2:<val>..."
    Returns:
      ids: (n,) int64
      meta: dict attr -> (max_id+1,) object array with values stored at meta[attr][id]
    """
    ids = np.empty(len(metadata_store), dtype=np.int64)

    # First pass: collect ids + all attributes
    parsed = []
    all_attrs = set()
    max_id = -1

    for i, s in enumerate(metadata_store):
        parts = s.split("__")
        # id:<int>
        k, v = parts[0].split(":", 1)
        if k != "id":
            raise ValueError(f"Expected first field 'id:<int>', got: {parts[0]}")
        _id = int(v)
        ids[i] = _id
        max_id = max(max_id, _id)

        kv = {"id": _id}
        for p in parts[1:]:
            if not p:
                continue
            ak, av = p.split(":", 1)
            kv[ak] = av
            all_attrs.add(ak)
        parsed.append(kv)

    # Allocate dense arrays by id (fast lookups in filter)
    meta: Dict[str, np.ndarray] = {}
    for attr in all_attrs:
        arr = np.empty((max_id + 1,), dtype=object)
        arr[:] = None
        meta[attr] = arr

    # Fill
    for kv in parsed:
        _id = kv["id"]
        for attr in all_attrs:
            meta[attr][_id] = kv.get(attr, None)

    return ids, meta

# ------------------------------- ABC ----------------------------------
# class ANNIndex:
#     """Abstract ANN index interface for a single subindex."""
#     def build(self, vectors: np.ndarray, ids: Sequence[int]) -> None:
#         raise NotImplementedError

#     def query(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
#         """Return (ids, distances) for top-k in this subindex."""
#         raise NotImplementedError


class FilterDiskANN:
    def __init__(
        self,
        args,
        d,
        metadata_store,
        protected_attrs: Dict[str, List[Any]],
        metric: str = "l2",
        min_bucket_size: int = 1,
    ) -> None:
        self.args = args
        self.metadata_store = metadata_store
        self.ids, self.meta = parse_metadata_store(metadata_store)
        self.max_id = int(self.ids.max())
        self.metric = metric
        self.min_bucket_size = min_bucket_size
        
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.index = hnswlib.Index(space=metric, dim=d)
        self._ef_construction = args.ef_construction
        self._M = args.M
        self.ef_search = args.ef_search


    def build_index(self, vectors: np.ndarray) -> None:
        ids_arr = np.asarray(self.ids, dtype=np.int64)
        self.index.init_index(
            max_elements=len(vectors),
            ef_construction=self._ef_construction,
            M=self._M,
        )
        self.index.add_items(vectors, ids_arr)
        self.index.set_ef(self.ef_search)


    def _build_eligible_mask(self, constraints):
        eligible = np.ones((self.max_id + 1,), dtype=np.bool_)
        allowed = {
            attr: set(count_dict.keys())
            for attr, count_dict in constraints.items()
        }
        for attr, allowed_vals in allowed.items():
            if attr not in self.meta:
                raise KeyError(f"Unknown attribute '{attr}'. Available: {list(self.meta.keys())}")
            allowed_list = list(set(allowed_vals))
            # eligible &= (self.meta[attr] in allowed_vals)
            eligible &= np.isin(self.meta[attr], allowed_list)

        return eligible


    def search_and_solve(self, query, vector_store, dfunc):
        q_vec = query["vector"] if "vector" in query else query["text_query_embedding"]
        max_candidates = self.args.filtering_multiplier * query['k']

        start = time.time()

        eligible = self._build_eligible_mask(query["count"])
        def filt(label: int) -> bool:
            return 0 <= label < eligible.shape[0] and bool(eligible[label])

        try:
            cands, dists = self.index.knn_query(q_vec, k=max_candidates, filter=filt)
        except RuntimeError:
            return None
        search_time = time.time() - start

        final_cands = [(self.metadata_store[oid], dist) for oid, dist in zip(cands[0], dists[0])]

        start = time.time()
        solver = build_solver(self.args)
        results = solver.solve(final_cands, query)
        postprocessing_time = time.time() - start
        
        results['search_time'] = search_time
        results['postprocessing_time'] = postprocessing_time
        results["total_scanned"] = 0

        return results