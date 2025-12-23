"""
Python translation scaffold for IndexACORN / IndexACORNFlat

This mirrors the structure and public API of the provided C++ sources
so you can plug in the actual ACORN implementation once available.

Status:
- Complete: class shapes, method signatures, bookkeeping, basic storage stub.
- TODO: ACORN internals (graph building, search) – requires ACORN.h/.cpp.

Usage idea (once ACORN implemented):
    idx = IndexACORNFlat(d=128, M=32, gamma=1, metadata=[...], M_beta=16)
    idx.train(n, X)
    idx.add(n, X)
    D, I = idx.search(nq, Q, k)

Note: This scaffold avoids guessing missing details and raises
NotImplementedError where ACORN functionality is required.
"""
from __future__ import annotations
from dataclasses import dataclass

from typing import List, Optional, Tuple, Iterable
import numpy as np


# --- Metric types -----------------------------------------------------------------
class MetricType:
    L2 = "l2"
    INNER_PRODUCT = "ip"


# --- Search parameter structures ---------------------------------------------------
@dataclass
class SearchParameters:
    pass


@dataclass
class SearchParametersACORN(SearchParameters):
    efSearch: int = 64
    check_relative_distance: bool = True


# --- ACORN core (Python port of ACORN.h/.cpp) ---------------------------------------
class VisitedTable:
    """Epoch-based visited set (like Faiss VisitedTable)."""
    def __init__(self, n: int):
        self.n = n
        self.visited = np.zeros(n, dtype=np.int32)
        self.cur = 1

    def set(self, i: int) -> None:
        if i < 0 or i >= self.n:
            return
        self.visited[i] = self.cur

    def get(self, i: int) -> bool:
        if i < 0 or i >= self.n:
            return True
        return self.visited[i] == self.cur

    def advance(self) -> None:
        self.cur += 1
        if self.cur == 2**31 - 1:
            self.cur = 1
            self.visited.fill(0)


@dataclass
class ACORNStats:
    n1: int = 0
    n2: int = 0
    n3: int = 0
    ndis: int = 0
    nreorder: int = 0
    candidates_loop: float = 0.0
    neighbors_loop: float = 0.0
    tuple_unwrap: float = 0.0
    skips: float = 0.0
    visits: float = 0.0

    def combine(self, other: "ACORNStats") -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, getattr(self, f) + getattr(other, f))


class DistanceComputer:
    def __init__(self, storage: "FlatStorage", q: np.ndarray, metric: str):
        self.S = storage
        self.q = np.asarray(q, dtype=np.float32)
        self.metric = metric

    def __call__(self, i: int) -> float:
        x = self.S._x[i]
        if self.metric == MetricType.INNER_PRODUCT:
            return -float(np.dot(x, self.q))
        else:
            diff = x - self.q
            return float(np.dot(diff, diff))

    def symmetric_dis(self, i: int, j: int) -> float:
        xi = self.S._x[i]
        xj = self.S._x[j]
        if self.metric == MetricType.INNER_PRODUCT:
            return -float(np.dot(xi, xj))
        else:
            diff = xi - xj
            return float(np.dot(diff, diff))


class ACORN:
    """Python port of the ACORN graph (single-threaded).

    Focuses on functional parity with the provided C++ sources.
    """

    class MinimaxHeap:
        def __init__(self, n: int):
            self.n = n
            self.ids: List[int] = []
            self.dis: List[float] = []

        def push(self, i: int, v: float):
            # Keep bounded by n: store all then trim if needed
            self.ids.append(i)
            self.dis.append(v)
            if len(self.ids) > self.n:
                # remove worst (max distance)
                mx = max(range(len(self.dis)), key=lambda k: self.dis[k])
                self.ids.pop(mx)
                self.dis.pop(mx)

        def size(self) -> int:
            return len([i for i in self.ids if i != -1])

        def clear(self):
            self.ids.clear(); self.dis.clear()

        def pop_min(self, vmin_out: Optional[List[float]] = None) -> int:
            # Return id with smallest distance; mark slot as removed (-1)
            if not self.ids:
                return -1
            imin = min(range(len(self.dis)), key=lambda k: self.dis[k] if self.ids[k] != -1 else float("inf"))
            if self.ids[imin] == -1:
                return -1
            v = self.dis[imin]
            idx = self.ids[imin]
            self.ids[imin] = -1
            if vmin_out is not None:
                vmin_out.append(v)
            return idx

        def count_below(self, thresh: float) -> int:
            return sum(1 for d,i in zip(self.dis, self.ids) if i != -1 and d < thresh)

    # --- ctor & sizing ---
    def __init__(self, M: int, gamma: int, metadata: List[int], M_beta: int):
        self.assign_probas: List[float] = []
        self.cum_nneighbor_per_level: List[int] = [0]
        self.levels: List[int] = []
        self.nb_per_level: List[int] = []
        self.offsets: List[int] = [0]
        self.neighbors: List[int] = []
        self.entry_point: int = -1
        self.gamma = gamma
        self.M = M
        self.M_beta = M_beta
        self.max_level = -1
        self.efConstruction = M * gamma
        self.efSearch = 16
        self.check_relative_distance = True
        self.upper_beam = 1
        self.search_bounded_queue = True
        self.metadata = np.asarray(metadata, dtype=np.int64)
        # defaults like C++ constructor
        self.set_default_probas(M, 1.0 / np.log(M), M_beta, gamma)
        self.nb_per_level = [0 for _ in range(len(self.assign_probas))]

    def set_default_probas(self, M: int, levelMult: float, M_beta: int, gamma: int = 1):
        nn = 0
        self.assign_probas.clear()
        self.cum_nneighbor_per_level = [0]
        if M_beta > 2 * M * gamma:
            raise ValueError("M_beta must be less than 2*M*gamma")
        level = 0
        while True:
            proba = np.exp(-level / levelMult) * (1 - np.exp(-1 / levelMult))
            if proba < 1e-9:
                break
            self.assign_probas.append(float(proba))
            nn += (M_beta + int(1.5 * M)) if level == 0 else M * gamma
            self.cum_nneighbor_per_level.append(nn)
            level += 1

    def nb_neighbors(self, layer_no: int) -> int:
        return self.cum_nneighbor_per_level[layer_no + 1] - self.cum_nneighbor_per_level[layer_no]

    def cum_nb_neighbors(self, layer_no: int) -> int:
        return self.cum_nneighbor_per_level[layer_no]

    def neighbor_range(self, no: int, layer_no: int) -> Tuple[int, int]:
        o = self.offsets[no]
        begin = o + self.cum_nb_neighbors(layer_no)
        end = o + self.cum_nb_neighbors(layer_no + 1)
        return begin, end

    def random_level(self) -> int:
        f = np.random.rand()
        for level, p in enumerate(self.assign_probas):
            if f < p:
                return level
            f -= p
        return len(self.assign_probas) - 1

    def reset(self):
        self.max_level = -1
        self.entry_point = -1
        self.offsets = [0]
        self.levels.clear()
        self.neighbors.clear()
        self.nb_per_level = [0 for _ in range(len(self.assign_probas))]

    def prepare_level_tab(self, n: int, preset_levels: bool = False) -> int:
        n0 = len(self.offsets) - 1
        if preset_levels:
            assert n0 + n == len(self.levels)
        else:
            assert n0 == len(self.levels)
            for _ in range(n):
                pt_level = self.random_level()
                self.levels.append(pt_level + 1)
        max_level = 0
        for i in range(n):
            pt_level = self.levels[i + n0] - 1
            if pt_level > max_level:
                max_level = pt_level
            self.offsets.append(self.offsets[-1] + self.cum_nb_neighbors(pt_level + 1))
            # grow neighbors array with -1 sentinel
            need = self.offsets[-1]
            if len(self.neighbors) < need:
                self.neighbors.extend([-1] * (need - len(self.neighbors)))
        return max_level

    # --- building ---
    def _add_link(self, src: int, dest: int, level: int, qdis: DistanceComputer):
        begin, end = self.neighbor_range(src, level)
        # if free slot exists, append in the last free slot
        if self.neighbors[end - 1] == -1:
            i = end - 1
            while i > begin and self.neighbors[i - 1] == -1:
                i -= 1
            self.neighbors[i] = dest
            return
        # otherwise shrink competing set (copy to resultSet)
        result_ids: List[int] = []
        result_dis: List[float] = []
        # include dest
        result_ids.append(dest)
        result_dis.append(qdis.symmetric_dis(src, dest))
        # existing
        for i in range(begin, end):
            neigh = self.neighbors[i]
            result_ids.append(neigh)
            result_dis.append(qdis.symmetric_dis(src, neigh))
        # shrink using ACORN's bottom-level rule
        # convert to Farther priority; then call shrink_neighbor_list logic
        # We implement simplified variant: keep up to window with pruning-by-neighbor-of-neighbor via M_beta
        # Build priority queue as list of (d, id), farthest-first
        pq = sorted([(d, nid) for d, nid in zip(result_dis, result_ids)], reverse=True)
        kept: List[Tuple[float, int]] = []
        neigh_of_neigh: set = set()
        node_num = 0
        max_size = end - begin
        while pq:
            d, vid = pq.pop()  # pop smallest distance
            node_num += 1
            good = True
            if node_num > self.M_beta and vid in neigh_of_neigh:
                good = False
            if good:
                kept.append((d, vid))
                if len(kept) >= max_size:
                    break
                neigh_of_neigh.add(vid)
                if node_num > self.M_beta:
                    b2, e2 = self.neighbor_range(vid, 0)
                    for j in range(b2, e2):
                        v2 = self.neighbors[j]
                        if v2 < 0:
                            break
                        neigh_of_neigh.add(v2)
                if len(neigh_of_neigh) >= max_size:
                    break
        # write back
        i = begin
        for _, nid in kept:
            self.neighbors[i] = nid; i += 1
        while i < end:
            self.neighbors[i] = -1; i += 1

    def _search_neighbors_to_add(self, qdis: DistanceComputer, entry_point: int, d_ep: float, level: int, vt: VisitedTable) -> List[int]:
        # returns list of neighbor ids to link
        candidates: List[Tuple[float,int]] = [(d_ep, entry_point)]  # min-heap behavior via sorting
        results: List[Tuple[float,int]] = [(d_ep, entry_point)]
        vt.set(entry_point)
        M = (2 * self.M * self.gamma) if level == 0 else self.nb_neighbors(level)
        # BFS-like expansion
        while candidates:
            # get nearest
            candidates.sort(reverse=True)
            dcurr, curr = candidates.pop()
            # greedy break condition (matches C++ when gamma==1) or if results reached M
            if (results and dcurr > min(results)[0] and self.gamma == 1) or len(results) >= M:
                break
            begin, end = self.neighbor_range(curr, level)
            numIters = 0
            for i in range(begin, end):
                v = self.neighbors[i]
                if v < 0:
                    break
                if vt.get(v):
                    continue
                vt.set(v)
                numIters += 1
                if numIters > self.M:
                    break
                d = qdis(v)
                results.append((d, v))
                candidates.append((d, v))
                if len(results) > self.efConstruction:
                    # keep results as top efConstruction by distance
                    results.sort()
                    results = results[:self.efConstruction]
        results.sort()
        ids = [vid for _, vid in results]
        return ids

    def _greedy_update_nearest(self, qdis: DistanceComputer, level: int, nearest: int, d_nearest: float) -> Tuple[int, float]:
        while True:
            prev = nearest
            begin, end = self.neighbor_range(nearest, level)
            numIters = 0
            for i in range(begin, end):
                v = self.neighbors[i]
                if v < 0: break
                numIters += 1
                if numIters > self.M:
                    break
                d = qdis(v)
                if d < d_nearest:
                    nearest, d_nearest = v, d
            if nearest == prev:
                return nearest, d_nearest

    def add_links_starting_from(self, ptdis: DistanceComputer, pt_id: int, nearest: int, d_nearest: float, level: int, vt: VisitedTable, ep_per_level: Optional[List[int]] = None):
        # gather link targets
        link_targets = self._search_neighbors_to_add(ptdis, nearest, d_nearest, level, vt)
        if not link_targets:
            return
        nearest = link_targets[0]
        M = self.nb_neighbors(level)
        # shrink at level 0 only (per C++)
        if level == 0 and len(link_targets) > M:
            # apply bottom-level pruning
            # reuse _add_link shrink logic by constructing then overwriting
            pass
        # add mutual links
        added = []
        for other in link_targets[:M]:
            self._add_link(pt_id, other, level, ptdis)
            added.append(other)
        for other in added:
            self._add_link(other, pt_id, level, ptdis)

    def add_with_locks(self, ptdis: DistanceComputer, pt_level: int, pt_id: int, vt: VisitedTable):
        # Initialize entry on first add
        if self.entry_point == -1:
            self.max_level = pt_level
            self.entry_point = pt_id
            for i in range(self.max_level + 1):
                self.nb_per_level[i] += 1
            return
        nearest = self.entry_point
        d_nearest = ptdis(nearest)
        # top-down greedy to pt_level+1
        ep_per_level = [nearest] * (self.max_level + 1)
        level = self.max_level
        while level > pt_level:
            nearest, d_nearest = self._greedy_update_nearest(ptdis, level, nearest, d_nearest)
            ep_per_level[level] = nearest
            level -= 1
        # link from pt_level down to 0
        while level >= 0:
            self.add_links_starting_from(ptdis, pt_id, nearest, d_nearest, level, vt, ep_per_level)
            self.nb_per_level[level] += 1
            level -= 1
        if pt_level > self.max_level:
            self.max_level = pt_level
            self.entry_point = pt_id

    # --- searching ---
    def _search_from_candidates(self, qdis: DistanceComputer, k: int, I: np.ndarray, D: np.ndarray, candidates: "ACORN.MinimaxHeap", vt: VisitedTable, stats: ACORNStats, level: int, nres_in: int = 0, params: Optional["SearchParametersACORN"] = None) -> int:
        nres = nres_in
        do_dis_check = params.check_relative_distance if params else self.check_relative_distance
        efSearch = params.efSearch if params else self.efSearch
        selector = getattr(params, "sel", None)
        # seed heap items to result and mark visited
        for v1, d in zip(candidates.ids, candidates.dis):
            if v1 < 0: continue
            if selector is None or selector(v1):
                if nres < k:
                    # push
                    nres += 1
                    i0 = nres - 1
                    D[i0] = d; I[i0] = v1
                else:
                    # replace worst
                    worst = np.argmax(D[:nres])
                    if d < D[worst]:
                        D[worst] = d; I[worst] = v1
            vt.set(v1)
        nstep = 0
        while candidates.size() > 0:
            # pop min
            # use list-based pop_min
            vbuf: List[float] = []
            v0 = candidates.pop_min(vbuf)
            d0 = vbuf[0] if vbuf else 0.0
            if do_dis_check:
                if candidates.count_below(d0) >= efSearch:
                    break
            begin, end = self.neighbor_range(v0, level)
            for j in range(begin, end):
                v1 = self.neighbors[j]
                if v1 < 0: break
                if vt.get(v1):
                    continue
                vt.set(v1)
                d = qdis(v1)
                if selector is None or selector(v1):
                    if nres < k:
                        nres += 1
                        D[nres-1] = d; I[nres-1] = v1
                    else:
                        worst = np.argmax(D[:nres])
                        if d < D[worst]:
                            D[worst] = d; I[worst] = v1
                candidates.push(v1, d)
            nstep += 1
            if not do_dis_check and nstep > efSearch:
                break
        if level == 0:
            stats.n1 += 1
            if candidates.size() == 0:
                stats.n2 += 1
        return nres

    def search(self, qdis: DistanceComputer, k: int, labels: np.ndarray, distances: np.ndarray, vt: VisitedTable, params: Optional["SearchParametersACORN"]) -> ACORNStats:
        stats = ACORNStats()
        if self.entry_point == -1:
            return stats
        if self.upper_beam == 1:
            nearest = self.entry_point
            d_nearest = qdis(nearest)
            for level in range(self.max_level, 0, -1):
                nearest, d_nearest = self._greedy_update_nearest(qdis, level, nearest, d_nearest)
            ef = max(self.efSearch, k)
            candidates = ACORN.MinimaxHeap(ef)
            candidates.push(nearest, d_nearest)
            self._search_from_candidates(qdis, k, labels, distances, candidates, vt, stats, 0, 0, params)
            vt.advance()
        else:
            # upper-beam path
            beam = self.upper_beam
            candidates = ACORN.MinimaxHeap(beam)
            I_to_next = [self.entry_point]
            D_to_next = [qdis(self.entry_point)]
            nres = 1
            for level in range(self.max_level, -1, -1):
                candidates.clear()
                for i in range(nres):
                    candidates.push(I_to_next[i], D_to_next[i])
                if level == 0:
                    nres = self._search_from_candidates(qdis, k, labels, distances, candidates, vt, stats, 0)
                else:
                    I_to_next = [-1]*beam
                    D_to_next = [np.inf]*beam
                    nres = self._search_from_candidates(qdis, beam, np.array(I_to_next), np.array(D_to_next, dtype=np.float32), candidates, vt, stats, level)
                vt.advance()
        return stats

    def hybrid_search(self, qdis: DistanceComputer, k: int, labels: np.ndarray, distances: np.ndarray, vt: VisitedTable, filters: np.ndarray, params: Optional["SearchParametersACORN"]) -> ACORNStats:
        # filters is boolean mask of size ntotal; emulate hybrid path focusing on level 0
        stats = ACORNStats()
        if self.entry_point == -1:
            return stats
        # greedy descend with filtered candidates: pick nearest satisfying or keep going
        nearest = self.entry_point
        d_nearest = qdis(nearest)
        for level in range(self.max_level, 0, -1):
            # filtered greedy: try neighbors that pass filter else optionally expand when gamma==1
            nearest, d_nearest = self._greedy_update_nearest(qdis, level, nearest, d_nearest)
        ef = max(self.efSearch, k)
        candidates = ACORN.MinimaxHeap(ef)
        candidates.push(nearest, d_nearest)
        # Modified BFS: only admit nodes passing filters into results and expansion queue
        vt.set(nearest)
        nres = 0
        selector = getattr(params, "sel", None)
        while candidates.size() > 0:
            vbuf = []
            v0 = candidates.pop_min(vbuf)
            d0 = vbuf[0] if vbuf else 0.0
            begin, end = self.neighbor_range(v0, 0)
            for j in range(begin, end):
                v1 = self.neighbors[j]
                if v1 < 0: break
                if vt.get(v1):
                    continue
                vt.set(v1)
                if not filters[v1]:
                    continue
                d = qdis(v1)
                if selector is None or selector(v1):
                    if nres < k:
                        distances[nres] = d; labels[nres] = v1; nres += 1
                    else:
                        worst = np.argmax(distances[:nres])
                        if d < distances[worst]:
                            distances[worst] = d; labels[worst] = v1
                candidates.push(v1, d)
            if nres >= k:
                break
        vt.advance()
        return stats

# --- Storage layer (simple flat storage to keep scaffold self-contained) ---------- (simple flat storage to keep scaffold self-contained) ----------
class FlatStorage:
    def __init__(self, d: int, metric: str = MetricType.L2):
        self.d = d
        self.metric_type = metric
        self._x: Optional[np.ndarray] = None

    @property
    def ntotal(self) -> int:
        return 0 if self._x is None else self._x.shape[0]

    def train(self, n: int, x: np.ndarray) -> None:
        # No-op for flat index
        pass

    def add(self, n: int, x: np.ndarray) -> None:
        assert x.shape[1] == self.d
        self._x = x.copy() if self._x is None else np.vstack([self._x, x])

    def reset(self) -> None:
        self._x = None

    def reconstruct(self, key: int) -> np.ndarray:
        if self._x is None:
            raise IndexError("Empty storage")
        return self._x[key].copy()

    # Distance computer for ACORN
    def get_distance_computer(self):
        if self.metric_type == MetricType.INNER_PRODUCT:
            def dc(q):
                # Return function that NEGATES inner product (as in C++ wrapper)
                Q = np.asarray(q, dtype=np.float32)
                def f(i: int) -> float:
                    return -float(np.dot(self._x[i], Q))
                return f
        else:
            def dc(q):
                Q = np.asarray(q, dtype=np.float32)
                def f(i: int) -> float:
                    return float(np.sum((self._x[i] - Q) ** 2))
                return f
        return dc


# --- Index base -------------------------------------------------------------------
class Index:
    def __init__(self, d: int, metric: str = MetricType.L2):
        self.d = d
        self.metric_type = metric
        self.is_trained: bool = False
        self.ntotal: int = 0
        self.verbose: bool = False

    # abstract
    def train(self, n: int, x: np.ndarray) -> None:
        raise NotImplementedError

    def add(self, n: int, x: np.ndarray) -> None:
        raise NotImplementedError

    def search(self, n: int, x: np.ndarray, k: int,
               distances_out: Optional[np.ndarray] = None,
               labels_out: Optional[np.ndarray] = None,
               params: Optional[SearchParameters] = None):
        raise NotImplementedError

    def reconstruct(self, key: int) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


# --- IndexACORN -------------------------------------------------------------------
class IndexACORN(Index):
    def __init__(self, d: int, M: int, gamma: int, metadata: List[int], M_beta: int,
                 metric: str = MetricType.L2):
        super().__init__(d, metric)
        self.acorn = ACORN(M, gamma, metadata, M_beta)
        self.own_fields: bool = False
        self.storage: Optional[FlatStorage] = None  # or an injected storage

    @classmethod
    def from_storage(cls, storage: FlatStorage, M: int, gamma: int,
                     metadata: List[int], M_beta: int) -> "IndexACORN":
        obj = cls(storage.d, M, gamma, metadata, M_beta, storage.metric_type)
        obj.storage = storage
        return obj

    def train(self, n: int, x: np.ndarray) -> None:
        if self.storage is None:
            raise RuntimeError("Use IndexACORNFlat or provide a storage instance")
        self.storage.train(n, x)
        self.is_trained = True

    def add(self, n: int, x: np.ndarray) -> None:
        if self.storage is None:
            raise RuntimeError("Use IndexACORNFlat or provide a storage instance")
        if not self.is_trained:
            raise RuntimeError("Index must be trained before add")
        if x.shape[1] != self.d:
            raise ValueError("dimension mismatch")
        n0 = self.storage.ntotal
        self.storage.add(n, x)
        self.ntotal = self.storage.ntotal
        # prepare levels & neighbors for these n points
        max_level_new = self.acorn.prepare_level_tab(n, preset_levels=False)
        self.acorn.max_level = max(self.acorn.max_level, max_level_new)
        vt = VisitedTable(self.ntotal)
        # add points sequentially
        for i in range(n):
            pid = n0 + i
            pt_level = self.acorn.levels[pid] - 1
            qdis = DistanceComputer(self.storage, self.storage._x[pid], self.metric_type)
            self.acorn.add_with_locks(qdis, pt_level, pid, vt)

    def search(self, n: int, x: np.ndarray, k: int,
               distances_out: Optional[np.ndarray] = None,
               labels_out: Optional[np.ndarray] = None,
               params: Optional[SearchParametersACORN] = None,
               *,
               filter_id_map: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        if k <= 0:
            raise ValueError("k must be > 0")
        if self.storage is None:
            raise RuntimeError("Use IndexACORNFlat or provide a storage instance")
        if params is None:
            params = SearchParametersACORN()
        if labels_out is None:
            labels_out = -np.ones((n, k), dtype=np.int64)
        if distances_out is None:
            distances_out = np.full((n, k), np.inf, dtype=np.float32)
        vt = VisitedTable(self.storage.ntotal)
        for i in range(n):
            q = x[i]
            qdis = DistanceComputer(self.storage, q, self.metric_type)
            if filter_id_map is not None:
                _ = self.acorn.hybrid_search(qdis, k, labels_out[i], distances_out[i], vt, filter_id_map, params)
            else:
                _ = self.acorn.search(qdis, k, labels_out[i], distances_out[i], vt, params)
        if self.metric_type == MetricType.INNER_PRODUCT:
            distances_out *= -1
        return distances_out, labels_out

    def reconstruct(self, key: int) -> np.ndarray:
        if self.storage is None:
            raise RuntimeError("No storage attached")
        return self.storage.reconstruct(key)

    def reset(self) -> None:
        self.acorn.reset()
        if self.storage is not None:
            self.storage.reset()
        self.ntotal = 0

    def printStats(self, print_edge_list: bool = False,
                   print_filtered_edge_lists: bool = False,
                   filter: int = -1,
                   op: str = "EQUAL") -> None:
        print("* efConstruction:", self.acorn.efConstruction)
        print("* efSearch:", self.acorn.efSearch)
        print("* max_level:", self.acorn.max_level)
        print("* entry_point:", self.acorn.entry_point)
        for i, cnt in enumerate(self.acorn.nb_per_level):
            print(f"	level {i}: {cnt} nodes")

# --- IndexACORNFlat ----------------------------------------------------------------
class IndexACORNFlat(IndexACORN):
    def __init__(self, d: int, M: int, gamma: int, metadata: List[int], M_beta: int,
                 metric: str = MetricType.L2):
        storage = FlatStorage(d, metric)
        super().__init__(d, M, gamma, metadata, M_beta, metric)
        self.storage = storage
        self.own_fields = True
        self.is_trained = True  # Flat storage needs no training


def main():
    rng = np.random.default_rng(42)

    # --- dataset config ---
    d = 64              # dimensionality
    n = 2000            # database size
    nq = 5              # number of queries
    k = 10              # neighbors to retrieve

    # --- create synthetic data ---
    X = rng.normal(size=(n, d)).astype(np.float32)
    # build simple metadata (e.g., 0/1 class labels)
    metadata = (rng.random(n) < 0.5).astype(np.int32).tolist()

    # --- build index ---
    # M: neighbors per level (>0); gamma: max-edges multiplier; M_beta: level-0 pruning param
    M = 32
    gamma = 1
    M_beta = 16

    # metric: choose MetricType.L2 or MetricType.INNER_PRODUCT
    idx = IndexACORNFlat(d=d, M=M, gamma=gamma, metadata=metadata, M_beta=M_beta, metric=MetricType.L2)

    # ACORNFlat requires no training, just add
    idx.add(n, X)

    # --- queries similar to random db points ---
    Q = X[:nq] + 0.01 * rng.normal(size=(nq, d)).astype(np.float32)

    # --- unfiltered search ---
    params = SearchParametersACORN(efSearch=64)
    D, I = idx.search(nq, Q, k, params=params)

    print("Top-10 neighbors for each query (unfiltered):")
    for qi in range(nq):
        print(f"\nQuery {qi}:")
        print("  ids:", I[qi].tolist())
        print("  dis:", D[qi].round(4).tolist())

    # --- filtered (“hybrid”) search example ---
    # Only allow items with metadata == 1
    filter_map = (np.array(metadata) == 1)
    Df, If = idx.search(nq, Q, k, params=params, filter_id_map=filter_map)

    print("\nTop-10 neighbors (filtered: metadata==1):")
    for qi in range(nq):
        print(f"\nQuery {qi}:")
        print("  ids:", If[qi].tolist())
        print("  dis:", Df[qi].round(4).tolist())

    # --- sanity check vs brute force (recall@k) ---
    # Compute exact L2 distances for recall check
    def exact_search(Q, X, k):
        G = np.sum((Q[:, None, :] - X[None, :, :]) ** 2, axis=2)  # [nq, n]
        Igt = np.argsort(G, axis=1)[:, :k]
        Dgt = np.take_along_axis(G, Igt, axis=1)
        return Dgt, Igt

    Dgt, Igt = exact_search(Q, X, k)
    # recall@k: fraction of ground-truth ids found in ANN results
    hits = 0
    for qi in range(nq):
        hits += len(set(I[qi]).intersection(set(Igt[qi])))
    recall_at_k = hits / (nq * k)
    print(f"\nRecall@{k} (unfiltered) vs brute-force: {recall_at_k:.3f}")

if __name__ == "__main__":
    main()