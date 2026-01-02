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

# ------------------------------- ABC ----------------------------------
class ANNIndex:
    """Abstract ANN index interface for a single subindex."""
    def build(self, vectors: np.ndarray, ids: Sequence[int]) -> None:
        raise NotImplementedError

    def query(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) for top-k in this subindex."""
        raise NotImplementedError


# ------------------------------- Main Classes ----------------------------------
class BruteForceIndex(ANNIndex):
    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self._vecs: Optional[np.ndarray] = None
        self._ids: Optional[np.ndarray] = None

    def build(self, vectors: np.ndarray, ids: Sequence[int]) -> None:
        self._vecs = np.asarray(vectors, dtype=np.float32)
        self._ids = np.asarray(ids, dtype=np.int64)

    def query(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._vecs is None or self._ids is None:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        q = _ensure_2d(np.asarray(query, dtype=np.float32))[0]
        if self.metric == "l2":
            d = l2_distances(q, self._vecs)
        else:
            d = cosine_distances(q, self._vecs)
        if len(d) == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
        k_eff = min(k, len(d))
        idx = np.argpartition(d, k_eff - 1)[:k_eff]
        idx_sorted = idx[np.argsort(d[idx])]
        return self._ids[idx_sorted], d[idx_sorted]


class HNSWIndex(ANNIndex):
    def __init__(self, dim: int, metric: str = "l2", ef_construction: int = 200, M: int = 16, ef_search: int = 64):
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'")
        space = "cosine" if metric == "cosine" else "l2"
        self.index = hnswlib.Index(space=space, dim=dim)
        self.ef_search = ef_search
        self.dim = dim
        self.metric = metric
        self._ids: Optional[np.ndarray] = None
        self._built = False
        self._ef_construction = ef_construction
        self._M = M

    def build(self, vectors: np.ndarray, ids: Sequence[int]) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        ids_arr = np.asarray(ids, dtype=np.int64)

        self.index.init_index(
            max_elements=len(vecs),
            ef_construction=self._ef_construction,
            M=self._M,
        )
        if len(vecs) > 0:
            self.index.add_items(vecs, ids_arr)

        self.index.set_ef(self.ef_search)
        self._built = True

    def query(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._built:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        cur = self.index.get_current_count()
        if cur == 0:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )

        # Clamp k to the number of elements in this subindex
        k_eff = min(k, cur)

        # Make sure ef_search is large enough for k_eff
        if self.ef_search < k_eff:
            self.index.set_ef(k_eff)
        q = _ensure_2d(np.asarray(query, dtype=np.float32))
        labels, distances = self.index.knn_query(q, k=k_eff)
        return labels[0].astype(np.int64), distances[0].astype(np.float32)


@dataclass
class CartesianKey:
    values: Tuple[Any, ...]
    def __hash__(self) -> int:
        return hash(self.values)
    def __str__(self) -> str:
        return ":".join(map(str, self.values))


class SIEVECartesian:
    """
    Manages a family of subindexes over the Cartesian product of attributes.

    Parameters
    ----------
    attr_names: list of attribute names to combine (e.g., ["gender", "race"]).
    attr_domains: dict str->list of values for each protected attribute.
    backend: "bruteforce" or "hnsw".
    metric: "cosine" or "l2".
    hnsw_params: dict passed to HNSWIndex if used.
    min_bucket_size: don't build subindex if its cardinality < min_bucket_size.
    """
    def __init__(
        self,
        args,
        d,
        metadata_store,
        protected_attrs: Dict[str, List[Any]],
        backend: str = "hnsw",
        metric: str = "l2",
        hnsw_params: Optional[Dict[str, Any]] = None,
        min_bucket_size: int = 1,
    ) -> None:
        self.args = args
        self._dim = d
        # self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.ids = [int(md.split('__')[0].split(':')[1]) for md in metadata_store]
        
        self.attr_domains = defaultdict(list)
        for item in protected_attrs:
            attr, value = item.split(':', 1)
            self.attr_domains[attr].append(value)
        
        self.attr_domains = {
            attr: list(dict.fromkeys(values))
            for attr, values in self.attr_domains.items()
        }
        self.attr_names = list(self.attr_domains.keys())

        ## get the names of all Cartesian attributes (e.g., gender:male__race:Hispanic...)
        groups = defaultdict(list)
        for a in protected_attrs:
            k, v = a.split(":", 1)
            groups[k].append(a)
        carte_attrs = list(itt.product(*groups.values()))
        self.all_carte_attrs = ["__".join(sorted(tup)) for tup in carte_attrs]

        self.partitions = self.group_ids_by_partition()
        self.partition_tokens = [
            (p, set(p.split("__")))
            for p in self.partitions
        ]
        self.backend = backend
        self.metric = metric
        self.hnsw_params = dict(
            ef_construction=args.ef_construction,  # or whatever you want
            M=args.M,
            ef_search=args.ef_search,
        )
        self.min_bucket_size = min_bucket_size

        # Storage
        self._subidx: Dict[CartesianKey, ANNIndex] = {}
        self._bucket_ids: Dict[CartesianKey, np.ndarray] = {}

    @staticmethod
    def parse_kv_string(s):
        """Parses strings like 'age:30-39__gender:female__race:indian' (or with id)
        into (id, features_dict). `id` is None if not present."""
        parts = s.split("__")
        kv = {}
        oid = None
        for p in parts:
            k, v = p.split(":", 1)
            k = k.strip().lower()
            v = v.strip().lower()
            if k == "id":
                oid = int(v)
            else:
                kv[k] = v
        return oid, kv

    @staticmethod
    def feature_key(feat_dict):
        """Order-agnostic canonical key for a feature dict (excluding id)."""
        return tuple(sorted(feat_dict.items()))

    def group_ids_by_partition(self):
        """
        partitions: list[str] like 'age:30-39__gender:female__race:indian'
        datapoints: list[str] like 'id:0__gender:male__age:50-59__race:east asian'
        returns: dict[str, list[int]] mapping the *original partition string* to ids
        """
        partitions = self.all_carte_attrs
        datapoints = self.metadata_store
        # 1) Build a lookup from features -> original partition string
        features_to_partition = {}
        out = {p: [] for p in partitions}  # ensure all partitions exist with empty lists
        for p in partitions:
            _, feats = self.parse_kv_string(p)
            features_to_partition[self.feature_key(feats)] = p
        # 2) Assign datapoint ids to matching partition
        for d in datapoints:
            oid, feats = self.parse_kv_string(d)
            if oid is None:
                continue  # or raise if an id is required
            key = self.feature_key(feats)
            part = features_to_partition.get(key)
            if part is not None:
                out[part].append(oid)
            # else: silently skip datapoints that don't match any partition
        return out

    # ------------------------- Fitting & Building -------------------------
    def build_index(self, vector_store) -> None:
        for key, ids_list in self.partitions.items():
            if len(ids_list) < self.min_bucket_size:
                # skip tiny buckets if requested
                continue
            ids_arr = np.asarray(ids_list, dtype=np.int64)
            vecs = vector_store[ids_arr]
            # Choose backend
            if self.backend == "hnsw":
                index = HNSWIndex(dim=self._dim, metric=self.metric, **self.hnsw_params)
            else:
                index = BruteForceIndex(metric=self.metric)
            index.build(vecs, ids_arr)
            self._subidx[key] = index
            self._bucket_ids[key] = ids_arr

    # ------------------------------ Querying ------------------------------
    def search_and_solve(
        self,
        query,
        dfunc=None,
        k_sub=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        constraints = query['count']
        query_vec = query['vector']
        k = int(query['k'])
        lists = [
            [f"{attr}:{value}" for value in values.keys()]
            for attr, values in constraints.items()
        ]

        # Cartesian product of fairness constraints
        carte_query = list(itt.product(*lists))
        # combo_info = {}
        final_cands = []
        total_scan = 0

        search_time = 0.
        postprocessing_time = 0.

        start = time.time()
        for combo in carte_query:
            counts = []
            for token in combo:
                attr, val = token.split(":", 1)
                counts.append(constraints[attr][val])
            requirement = min(counts)

            # --- matching partitions containing all tokens in combo ---
            combo_set = set(combo)
            matching_parts = [
                p for p, tokens in self.partition_tokens
                if combo_set.issubset(tokens)
            ]
            # keep only keys we actually built
            matching_parts = [part_ for part_ in matching_parts if part_ in self._subidx]

            # combo_info[combo] = {
            #     "requirement": requirement,
            #     "partitions": matching_parts,
            # }

            # if not mkeys:
            #     # No subindex built for these combos -> fall back to brute-force over all that match
            #     mask = np.ones(len(self._ids), dtype=bool)
            #     for name, want in attr_filter.items():
            #         vals = {want} if not isinstance(want, (set, frozenset)) else set(want)
            #         has = np.array([self._attrs[int(i)][name] in vals for i in self._ids], dtype=bool)
            #         mask &= has
            #     base_ids = self._ids[mask]
            #     base_vecs = self._vectors[mask]
            #     # Use BruteForce for the filtered pool
            #     bf = BruteForceIndex(metric=self.metric)
            #     bf.build(base_vecs, base_ids)
            #     return bf.query(query_vec, k)


            # m = len(matching_parts)
            # if k_sub is None:
                # Heuristic: proportional over-fetch per bucket
                # k_sub = max(10, math.ceil(requirement / max(1, m)) * 2)

            k_sub = requirement

            # Gather candidates from each subindex
            cand_ids: List[int] = []
            cand_dists: List[float] = []
            # print(matching_parts)
            for key in matching_parts:
                idx = self._subidx[key]
                ids, dists = idx.query(query_vec, k_sub)
                cand_ids.extend(map(int, ids.tolist()))
                cand_dists.extend(dists.tolist())

                total_scan += len(ids)

            # Deduplicate by best distance per id
            best: Dict[int, float] = {}
            for i, dist in zip(cand_ids, cand_dists):
                if i not in best or dist < best[i]:
                    best[i] = dist

            if not best:
                return None

            all_ids = np.fromiter(best.keys(), dtype=np.int64, count=len(best))
            all_dists = np.fromiter(best.values(), dtype=float, count=len(best))

            k_eff = min(requirement, len(all_ids))

            pairs = sorted(zip(all_ids, all_dists), key=lambda x: x[1])
            per_partition_cands = [(obj_id, dist) for obj_id, dist in pairs[:k_eff]]
            final_cands.extend(per_partition_cands)

        search_time += time.time() - start

        final_cands = [(self.metadata_store[cand[0]], cand[1]) for cand in final_cands]

        start = time.time()
        solver = build_solver(self.args)
        results = solver.solve(final_cands, query)
        postprocessing_time += time.time() - start
        
        results['search_time'] = search_time
        results['postprocessing_time'] = postprocessing_time
        results["total_scanned"] = total_scan

        return results