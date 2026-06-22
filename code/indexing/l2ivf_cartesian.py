"""
l2ivf_cartesian.py
------------------
IVF retrieval backend for MAFair-KNN — Design A: one FAISS IVF index PER
Cartesian-attribute partition. This is the IVF counterpart of l2lsh_cartesian.py
(which builds one LSH index per partition). Everything else — Cartesian
partitioning, bitmap partition matching, per-partition quota k_pi, and the
fair-selection post-processing (Post-Alg-1/2/3+) — is inherited unchanged,
because post-processing is agnostic to how the candidate pool C is produced.

Only the within-partition near-neighbour search changes:
    LSH:  probe ell hash tables, gather colliding ids.
    IVF:  probe `nprobe` of `nlist` Voronoi cells (k-means coarse quantizer),
          return the approximate top-k_star ids.

The IVF "effort" knob is `nprobe` (query-time, no rebuild) — the analogue of
LSH's `ell`. `nlist` (number of cells) is a build-time parameter; by default
nlist_pi = round(4*sqrt(n_pi)) capped so each cell has >= ~39 training points
(FAISS guidance). Tiny partitions fall back to a flat (brute-force) index.

Cache: one `<partition>.faissindex` per partition on disk (use_cache=True),
keyed by nlist/seed/metric so it is reused across nprobe values.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import math
import time
import numpy as np
import tqdm
import faiss

from l2lsh_cartesian import (
    L2LSHCartesian, reconstruct_metadata, ATTR_NAMES, ATTR_VALUES,
)
from solver import build_solver

_FLAT_THRESHOLD = 64        # partitions smaller than this use a flat index
                            # (below this IVF can't train >=2 cells meaningfully)


class L2IVFCartesian(L2LSHCartesian):
    def __init__(self, args, d, metadata_store, protected_attrs, attrs_memmap=None):
        # ---- replicate the partition setup from the LSH index (no hashing) ----
        self.args = args
        self.d = d
        self.attrs_memmap = attrs_memmap
        self.metadata_store = metadata_store
        self.c = float(args.c)
        self.r = float(getattr(args, "r", 0.0) or 0.0)
        self.w = float(getattr(args, "w", 0.0) or 0.0)
        self.delta = float(args.delta)
        self.rng = np.random.default_rng(args.seed)

        # IVF params
        self.nlist_arg = int(getattr(args, "nlist", 0))      # 0 => auto
        self.nprobe = max(1, int(getattr(args, "nprobe", 8)))
        self.oversample = int(getattr(args, "oversample", 100))
        self.metric = (faiss.METRIC_INNER_PRODUCT
                       if args.fdist == "cosine" else faiss.METRIC_L2)

        import itertools as itt
        from collections import defaultdict
        groups = defaultdict(list)
        for a in protected_attrs:
            kk, vv = a.split(":", 1)
            groups[kk].append(a)
        carte_attrs = list(itt.product(*groups.values()))
        self.all_carte_attrs = ["__".join(sorted(tup)) for tup in carte_attrs]

        self.partitions = self.group_ids_by_partition()      # inherited
        self.partition_tokens = [(p, set(p.split("__"))) for p in self.partitions]

        self.tables = {}        # pi -> faiss index (in-memory) OR path (cached)
        self._loaded = {}       # pi -> faiss index (lazy-loaded cache during query)

    # ---- cache layout ----
    def _cache_dir(self) -> str:
        nl = "auto" if self.nlist_arg == 0 else str(self.nlist_arg)
        return os.path.join(
            self.args.data_dir,
            f"ivf_cache_nlist={nl}_metric={self.metric}_seed={self.args.seed}",
        )

    def _cache_path(self, pi: str) -> str:
        safe = pi.replace(":", "_").replace("__", "-")
        return os.path.join(self._cache_dir(), safe + ".faissindex")

    def _cache_exists(self, pi: str) -> bool:
        return os.path.exists(self._cache_path(pi))

    def _shelve_exists(self, pi: str) -> bool:   # name kept for parity
        return self._cache_exists(pi)

    def _nlist_for(self, n_pi: int) -> int:
        if self.nlist_arg > 0:
            nl = self.nlist_arg
        else:
            nl = int(round(4 * math.sqrt(max(1, n_pi))))
        # FAISS wants >= ~39 training points per centroid
        nl = max(1, min(nl, max(1, n_pi // 39)))
        return nl

    def _build_partition(self, ids_arr: np.ndarray, X) -> faiss.Index:
        """Build one IVF (or flat) index over the partition's vectors."""
        n_pi = len(ids_arr)
        d = self.d
        if n_pi < _FLAT_THRESHOLD:
            base = faiss.IndexFlatL2(d) if self.metric == faiss.METRIC_L2 \
                else faiss.IndexFlatIP(d)
            index = faiss.IndexIDMap(base)
            Xp = np.ascontiguousarray(X[ids_arr], dtype=np.float32)
            if self.metric == faiss.METRIC_INNER_PRODUCT:
                faiss.normalize_L2(Xp)
            index.add_with_ids(Xp, ids_arr.astype(np.int64))
            return index

        nlist = self._nlist_for(n_pi)
        quantizer = (faiss.IndexFlatIP(d) if self.metric == faiss.METRIC_INNER_PRODUCT
                     else faiss.IndexFlatL2(d))
        index = faiss.IndexIVFFlat(quantizer, d, nlist, self.metric)

        # train on a (bounded) sample, then add everything in chunks
        n_train = min(n_pi, max(39 * nlist, 50_000))
        sample_ids = ids_arr if n_pi == n_train else \
            ids_arr[np.linspace(0, n_pi - 1, n_train).astype(np.int64)]
        Xtrain = np.ascontiguousarray(X[sample_ids], dtype=np.float32)
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(Xtrain)
        index.train(Xtrain)

        CH = 1_000_000
        for s in range(0, n_pi, CH):
            e = min(s + CH, n_pi)
            ids_chunk = ids_arr[s:e]
            Xc = np.ascontiguousarray(X[ids_chunk], dtype=np.float32)
            if self.metric == faiss.METRIC_INNER_PRODUCT:
                faiss.normalize_L2(Xc)
            index.add_with_ids(Xc, ids_chunk.astype(np.int64))
        return index

    def build_index(self, X, chunk_size=10_000, n_workers=1, use_cache=False):
        assert X.shape[1] == self.d, "Dimension mismatch in build_index"
        if use_cache:
            os.makedirs(self._cache_dir(), exist_ok=True)
        for pi, ids in tqdm.tqdm(self.partitions.items(), desc="Indexing partitions (IVF)"):
            if not len(ids):
                continue
            if use_cache and self._cache_exists(pi):
                print(f"[build_index] cached — skipping '{pi}'", flush=True)
                self.tables[pi] = self._cache_path(pi)
                continue
            ids_arr = np.sort(np.asarray(ids, dtype=np.int64))
            index = self._build_partition(ids_arr, X)
            if use_cache:
                faiss.write_index(index, self._cache_path(pi))
                self.tables[pi] = self._cache_path(pi)
                del index
            else:
                self.tables[pi] = index

    def _get_index(self, pi) -> faiss.Index:
        """Return the partition's faiss index, lazy-loading + caching if on disk."""
        t = self.tables[pi]
        if isinstance(t, str):
            idx = self._loaded.get(pi)
            if idx is None:
                idx = faiss.read_index(t)
                self._loaded[pi] = idx
            return idx
        return t

    # ---- query ----
    def search_and_solve(self, query, vector_store, dfunc, c_min=None):
        import itertools as itt
        final_cands = []
        data = query['count']
        q = query.get('text_query_embedding', query.get('vector'))
        qf = np.ascontiguousarray(np.asarray(q, dtype=np.float32).reshape(1, -1))
        if self.metric == faiss.METRIC_INNER_PRODUCT:
            faiss.normalize_L2(qf)

        lists = [[f"{attr}:{v}" for v in vals.keys()] for attr, vals in data.items()]
        carte_query = list(itt.product(*lists))

        combo_info = {}
        for combo in carte_query:
            counts = []
            for token in combo:
                attr, val = token.split(":", 1)
                constraint = data[attr][val]
                ub = constraint[1] if isinstance(constraint, (tuple, list)) else constraint
                counts.append(ub)
            combo_set = set(combo)
            matching = [p for p, toks in self.partition_tokens if combo_set.issubset(toks)]
            combo_info[combo] = {"requirement": min(counts), "partitions": matching}

        n_combos = len(combo_info)
        total_scan = 0
        search_time = 0.
        start = time.time()
        for combo, info in combo_info.items():
            k_pi = info['requirement']
            if c_min is not None:
                k_pi = max(k_pi, math.ceil(c_min / n_combos))
            k_star = k_pi + self.oversample

            all_cands = []
            for pi in info["partitions"]:
                index = self._get_index(pi)
                try:
                    index.nprobe = self.nprobe      # IVF only; flat has no nprobe
                except AttributeError:
                    pass
                _, I = index.search(qf, k_star)
                cands = [int(x) for x in I[0] if x >= 0]
                all_cands.extend(cands)
                total_scan += len(cands)

            all_cands = list(set(all_cands))
            all_dists = [dfunc(q, vector_store[i]) for i in all_cands]
            pairs = sorted(zip(all_cands, all_dists), key=lambda x: x[1])
            final_cands.extend([(oid, dist) for oid, dist in pairs[:k_pi]])
        search_time += time.time() - start

        # map ids -> metadata strings (same as LSH path)
        if self.attrs_memmap is not None:
            final_cands = [(reconstruct_metadata(int(c[0]), self.attrs_memmap[int(c[0])]), c[1])
                           for c in final_cands]
        else:
            final_cands = [(self.metadata_store[c[0]], c[1]) for c in final_cands]

        seen, deduped = set(), []
        for meta, dist in sorted(final_cands, key=lambda x: x[1]):
            oid = meta.split("__")[0]
            if oid not in seen:
                seen.add(oid); deduped.append((meta, dist))
        final_cands = deduped

        if not len(final_cands):
            return None

        start = time.time()
        if self.args.m > 1:
            results = build_solver(self.args).solve(final_cands, query)
        else:
            count = {k: {v: 0 for v in reqs} for k, reqs in query['count'].items()}
            for text, _ in final_cands:
                for part in text.split("__"):
                    if ":" not in part:
                        continue
                    key, val = part.split(":")
                    if key in query['count'] and val in query['count'][key]:
                        count[key][val] += 1
            results = {'objective': sum(c[1] for c in final_cands),
                       'selected': [c[0] for c in final_cands], 'count': count}
        postprocessing_time = time.time() - start

        results['search_time'] = search_time
        results['postprocessing_time'] = postprocessing_time
        results['total_scanned'] = total_scan
        results['pool_size'] = len(final_cands)
        return results
