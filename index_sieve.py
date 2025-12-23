"""
SIEVE (Set of Indexes for Efficient Vector Exploration) — minimal Python implementation

High-level idea (matching the paper and OSS repo semantics):
- Build a *collection* of sub-indexes on frequent attribute values (per-attribute postings),
  plus a global fallback index. 
- At query time with filters, choose a small subset of sub-indexes that covers the filter
  (union over selected postings), search them, and merge results. If the cost estimate is
  high (too many postings), fallback to global index + post-filtering.

This implementation targets *categorical* filters (equality or IN lists). It can be extended
for range filters by pre-bucketing.

Dependencies: requires `index_acorn.py` in the same folder (the ACORN port from the canvas).

NOTE: This is a faithful *functional* model of SIEVE's core mechanism without the full
workload-driven optimizer from the paper. The selection rule is a simple greedy heuristic
that mirrors the paper's spirit while staying compact.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Set
import numpy as np

from index_acorn import IndexACORNFlat, MetricType, SearchParametersACORN


@dataclass
class Posting:
    """A postings entry: a sub-index over a subset of points."""
    name: str
    ids: np.ndarray  # global ids covered by this posting
    index: IndexACORNFlat


@dataclass
class SIEVEConfig:
    metric: str = MetricType.L2
    M: int = 32
    gamma: int = 1
    M_beta: int = 16
    freq_threshold: float = 0.05  # build sub-index when value freq >= threshold
    efSearch: int = 64
    # Fallback policy: when covered candidate pool exceeds this fraction of N, use global
    fallback_frac: float = 0.5


class SIEVEIndex:
    def __init__(self, d: int, config: Optional[SIEVEConfig] = None):
        self.d = d
        self.cfg = config or SIEVEConfig()
        self.global_index = IndexACORNFlat(d, self.cfg.M, self.cfg.gamma, metadata=[], M_beta=self.cfg.M_beta, metric=self.cfg.metric)
        self.N: int = 0
        # attribute-> value -> Posting
        self.postings: Dict[str, Dict[int, Posting]] = {}
        # Attribute value arrays (global ids)
        self.attrs: Dict[str, np.ndarray] = {}

    # ----------------------------- BUILD -------------------------------------
    def add(self, X: np.ndarray, attrs: Dict[str, np.ndarray]):
        """Add a full batch of vectors and attribute columns.

        X: (N, d) float32
        attrs: mapping attr_name -> int array of length N (categorical codes)
        """
        assert X.ndim == 2 and X.shape[1] == self.d
        N = X.shape[0]
        for k, col in attrs.items():
            assert len(col) == N, f"attr {k} length mismatch"
        self.N = N
        self.attrs = {k: np.asarray(v, dtype=np.int64) for k, v in attrs.items()}

        # Build global index
        self.global_index.add(N, X.astype(np.float32))

        # Build postings for frequent values
        for attr, col in self.attrs.items():
            self.postings[attr] = {}
            # value counts
            vals, counts = np.unique(col, return_counts=True)
            for v, c in zip(vals, counts):
                freq = c / N
                if freq >= self.cfg.freq_threshold and c >= max(50, self.cfg.M*2):
                    mask = (col == v)
                    ids = np.nonzero(mask)[0].astype(np.int64)
                    # metadata for sub-index can be the attribute value itself; not used in search
                    metadata = (np.ones(c, dtype=np.int32) * int(v)).tolist()
                    sub = IndexACORNFlat(self.d, self.cfg.M, self.cfg.gamma, metadata, self.cfg.M_beta, self.cfg.metric)
                    sub.add(c, X[ids].astype(np.float32))
                    self.postings[attr][int(v)] = Posting(name=f"{attr}={int(v)}", ids=ids, index=sub)

    # --------------------------- QUERY ----------------------------------------
    def _select_postings(self, filters: Dict[str, Iterable[int]]) -> List[Posting]:
        """Greedy covering over attribute=value postings.
        Returns a list of postings to query (union semantics).
        If no postings match or the candidate pool is too big, returns empty list to signal fallback.
        """
        if not filters:
            return []
        candidate_ids: Set[int] = set()
        chosen: List[Posting] = []
        for attr, values in filters.items():
            values = list(values)
            if attr not in self.postings:
                # No prebuilt postings for this attr -> signal fallback
                return []
            # pick all postings that exist for requested values
            exist = [self.postings[attr].get(int(v)) for v in values]
            exist = [p for p in exist if p is not None]
            if not exist:
                # None exist -> fallback
                return []
            # Greedy: sort by posting size ascending to prioritize tighter indexes
            exist.sort(key=lambda p: p.ids.size)
            for p in exist:
                # union because filter is attr IN values
                chosen.append(p)
                candidate_ids.update(p.ids.tolist())
        # If the union is too large, fallback to global
        if len(candidate_ids) > self.cfg.fallback_frac * self.N:
            return []
        return chosen

    def search(self, Q: np.ndarray, k: int, filters: Optional[Dict[str, Iterable[int]]] = None, params: Optional[SearchParametersACORN] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Search with optional attribute filters.

        Returns (D, I) with global ids.
        """
        assert Q.ndim == 2 and Q.shape[1] == self.d
        nq = Q.shape[0]
        params = params or SearchParametersACORN(efSearch=self.cfg.efSearch)

        # 1) try postings-based plan
        postings = self._select_postings(filters or {})
        if postings:
            # Per-posting searches; then merge
            all_I: List[np.ndarray] = []
            all_D: List[np.ndarray] = []
            for p in postings:
                Dp, Ip = p.index.search(nq, Q.astype(np.float32), k, params=params)
                # map local ids back to global
                Ip_global = np.where(Ip >= 0, p.ids[Ip], -1)
                all_I.append(Ip_global)
                all_D.append(Dp)
            # merge across postings (per query)
            D = np.full((nq, k), np.inf, dtype=np.float32)
            I = -np.ones((nq, k), dtype=np.int64)
            for qi in range(nq):
                # concatenate candidates
                candI = np.concatenate([Iarr[qi] for Iarr in all_I], axis=0)
                candD = np.concatenate([Darr[qi] for Darr in all_D], axis=0)
                # dedup by global id, keep best distance
                best: Dict[int, float] = {}
                for gid, d in zip(candI.tolist(), candD.tolist()):
                    if gid < 0:
                        continue
                    if gid not in best or d < best[gid]:
                        best[gid] = d
                if not best:
                    continue
                # keep top-k
                items = sorted(best.items(), key=lambda t: t[1])[:k]
                I[qi, :len(items)] = [t[0] for t in items]
                D[qi, :len(items)] = [t[1] for t in items]
            # final post-filter to guarantee correctness for requested values
            if filters:
                mask = self._build_filter_mask()
                ok = self._eval_filters(mask, filters)
                for qi in range(nq):
                    for j in range(k):
                        gid = I[qi, j]
                        if gid >= 0 and not ok[gid]:
                            I[qi, j] = -1; D[qi, j] = np.inf
                # re-compact rows
                for qi in range(nq):
                    order = np.argsort(D[qi])
                    I[qi] = I[qi, order]
                    D[qi] = D[qi, order]
            return D, I

        # 2) fallback: global index + boolean map filtering
        filter_map: Optional[np.ndarray] = None
        if filters:
            mask = self._build_filter_mask()
            ok = self._eval_filters(mask, filters)
            filter_map = ok.astype(bool)
        D, I = self.global_index.search(nq, Q.astype(np.float32), k, params=params, filter_id_map=filter_map)
        return D, I

    # ---------------------- FILTER MASK HELPERS --------------------------------
    def _build_filter_mask(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.attrs.items()}

    @staticmethod
    def _eval_filters(mask: Dict[str, np.ndarray], filters: Dict[str, Iterable[int]]) -> np.ndarray:
        """Return boolean mask over global ids that satisfy all filters (AND over attrs, IN over values)."""
        N = len(next(iter(mask.values()))) if mask else 0
        ok = np.ones(N, dtype=bool)
        for attr, values in (filters or {}).items():
            values = set(int(v) for v in values)
            col = mask.get(attr)
            if col is None:
                ok &= False
            else:
                ok &= np.isin(col, list(values))
        return ok


# ----------------------------- DEMO ---------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    d = 64
    N = 10000
    X = rng.normal(size=(N, d)).astype(np.float32)
    # Two categorical attributes
    country = rng.integers(0, 5, size=N, dtype=np.int64)  # 5 countries
    topic = rng.integers(0, 20, size=N, dtype=np.int64)   # 20 topics

    sieve = SIEVEIndex(d)
    sieve.add(X, {"country": country, "topic": topic})

    Q = X[:3] + 0.01 * rng.normal(size=(3, d)).astype(np.float32)
    params = SearchParametersACORN(efSearch=80)

    print("Unfiltered search:")
    D, I = sieve.search(Q, k=10, params=params)
    print(I[0][:10])

    print("Filtered (country in {1,3}, topic in {2,4,6}):")
    D2, I2 = sieve.search(Q, k=10, filters={"country": [1,3], "topic": [2,4,6]}, params=params)
    print(I2[0][:10])
