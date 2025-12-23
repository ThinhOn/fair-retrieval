from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import math
import time
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Dict, List, Iterable, Optional, Callable, Any
from collections import defaultdict
import itertools as itt
from scipy.stats import norm
from scipy.integrate import quad
from solver import build_solver

import faiss

HashKey = Tuple[int, ...]  # the concatenated (k-long) hash key


# -----------------------------
# Base LSH families (single hash)
# -----------------------------

@dataclass
class L2Hash:  # p-stable LSH for L2 (Datar et al., 2004)
    a: np.ndarray      # shape (d,)
    b: float           # offset in [0, w)
    w: float           # bucket width

    @staticmethod
    def sample(d: int, w: float, rng: np.random.Generator) -> "L2Hash":
        a = rng.normal(size=d)         # Gaussian ~ N(0, I)
        b = rng.uniform(0.0, w)
        return L2Hash(a=a, b=b, w=w)

    def __call__(self, x: np.ndarray) -> int:
        return int(np.floor((self.a @ x + self.b) / self.w))

# ------------------------------------------------------
# Compound hash g(x) = (h1(x), ..., hk(x)) for one table
# ------------------------------------------------------

@dataclass
class CompoundHash:
    funcs: Tuple[Callable[[np.ndarray], int], ...]  # length k
    A: Optional[np.ndarray] = None   # (k, d) for L2
    b: Optional[np.ndarray] = None   # (k,)
    w: Optional[float] = None        # bucket width (same for all funcs in L2)

    def __post_init__(self):
        # If funcs are L2Hash objects, pre-pack into matrices for fast batch hashing
        if self.funcs and isinstance(self.funcs[0], L2Hash):
            As = []
            bs = []
            w = self.funcs[0].w
            for f in self.funcs:
                assert isinstance(f, L2Hash), "Mixed hash types not supported in vectorized path"
                As.append(f.a)
                bs.append(f.b)
                assert f.w == w, "All L2Hash in a compound family must share the same w"
            self.A = np.stack(As, axis=0)      # (k, d)
            self.b = np.asarray(bs)            # (k,)
            self.w = w

    def __call__(self, x: np.ndarray) -> HashKey:
        """
        Hash a single vector x. Keeps old API: returns a tuple[int,...].
        Uses vectorized path if available.
        """
        if self.A is not None:
            # (k,d) @ (d,) -> (k,)
            proj = (self.A @ x + self.b) / self.w
            return tuple(np.floor(proj).astype(int).tolist())
        else:
            return tuple(f(x) for f in self.funcs)

    def batch(self, X: np.ndarray) -> np.ndarray:
        """
        Hash a batch of vectors.
        X: (n, d)
        Returns: (n, k) int64 array of hash coordinates.
        """
        if self.A is not None:
            # X: (n,d), A: (k,d) -> (n,k)
            proj = (X @ self.A.T + self.b) / self.w
            return np.floor(proj).astype(np.int64)
        else:
            # Generic (slower) fallback if funcs are not L2Hash
            n = X.shape[0]
            k = len(self.funcs)
            out = np.empty((n, k), dtype=np.int64)
            for j, f in enumerate(self.funcs):
                # Apply scalar hash to each row
                out[:, j] = np.apply_along_axis(f, 1, X)
            return out
            
def make_compound_family(
    family_sampler: Callable[[], Callable[[np.ndarray], int]],
    mu: int,
    rng: np.random.Generator,
) -> CompoundHash:
    funcs = tuple(family_sampler() for _ in range(mu))
    return CompoundHash(funcs)



# -----------------------------------------
# Multi-table LSH index (ℓ independent g_j)
# -----------------------------------------

class L2LSHJoint:
    def __init__(
        self,
        args,
        d,
        metadata_store,
        protected_attrs,
    ):
        self.args = args
        self.d = d
        self.metadata_store = metadata_store
        self.c = float(args.c)
        self.r = float(args.r)
        self.w = float(args.w)
        self.delta = float(args.delta)
        self.rng = np.random.default_rng(args.seed)

        self.proportions = {
            attr: len([m for m in metadata_store if attr in m]) / len(metadata_store)
            for attr in protected_attrs
        }

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
        # print(f"example partition name: {list(self.partitions.keys())[0]}")

        self.tables = {}
        self.hashes = {}
        p1 = self.collision_probability(self.r, self.w)
        p2 = self.collision_probability(self.c * self.r, self.w)
        for pi, oids in self.partitions.items():    # for each partition \pi
            if args.mu == 0:
                mu = math.ceil( math.log(len(oids)) / math.log(1/p2) )
                mu = max(1, mu)
            else:
                mu = args.mu
            # ell = math.ceil( math.log(self.delta) / math.log(1 - p1**mu) )
            ell = args.ell
            self.tables[pi] = [defaultdict(list) for _ in range(ell)]
            self.hashes[pi] = []

            for _ in range(ell):
                # if family == "l2":
                def sampler() -> Callable[[np.ndarray], int]:
                    h = L2Hash.sample(d=self.d, w=self.w, rng=self.rng)
                    return h
                # else:
                #     def sampler() -> Callable[[np.ndarray], int]:
                #         h = HyperplaneHash.sample(d=self.d, rng=self.rng)
                #         return h
                self.hashes[pi].append(make_compound_family(sampler, mu=mu, rng=self.rng))
            # print(mu, ell)
            # print(self.hashes)
            # exit()

    @staticmethod
    def collision_probability(r, w):
        """
        Compute collision probability for L2 LSH given distance r and bucket width w.
        """
        # integrand: PDF of N(0, r^2) scaled + linear term (1 - t/w)
        integrand = lambda t: (1 - t / w) * (1 / r) * norm.pdf(t / r)
        result, _ = quad(integrand, 0, w)
        return 2 * result  # account for both sides of distribution

    @staticmethod
    def parse_kv_string(s):
        """Parses strings like 'age:30-39__gender:female__race:indian' (or with id)
        into (id, features_dict). `id` is None if not present."""
        parts = s.split("__")
        kv = {}
        sid = None
        for p in parts:
            k, v = p.split(":", 1)
            k = k.strip().lower()
            v = v.strip().lower()
            if k == "id":
                sid = int(v)
            else:
                kv[k] = v
        return sid, kv

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
            sid, feats = self.parse_kv_string(d)
            if sid is None:
                continue  # or raise if an id is required
            key = self.feature_key(feats)
            part = features_to_partition.get(key)
            if part is not None:
                out[part].append(sid)
            # else: silently skip datapoints that don't match any partition
        out = {k : v for k, v in out.items() if len(v)}
        return out


    # ---- indexing ----
    # def hash_point(self, vec, meta):
    #     """
    #     Old per-point API (still works, but is slow for large n).
    #     Build buckets for a single data vector.
    #     """
    #     meta = meta.split('__')
    #     object_id = int(meta[0].split(':')[1])
    #     pi = "__".join(sorted(meta[1:]))
    #     assert pi in self.partitions, "Error: partition not found, check data point processing!!"
    #     for T, g in zip(self.tables[pi], self.hashes[pi]):
    #         T[g(vec)].append(object_id)

    def build_index(self, X: np.ndarray):
        """
        NEW: fast bulk indexing.
        X: (n, d), where row i corresponds to object id i (as used in metadata_store).
        Uses precomputed partitions and vectorized hashing.
        """
        assert X.shape[1] == self.d, "Dimension mismatch in build_index"

        # For each partition pi, we know exactly which ids belong to it.
        for pi, ids in self.partitions.items():
            # print(f"Indexing partition {pi} with {len(ids)} points, ell = {len(self.tables[pi])}...")
            if not ids:
                continue
            ids_arr = np.asarray(ids, dtype=np.int64)
            X_pi = X[ids_arr]          # (m, d) subset for this partition

            # For each table in this partition, hash the whole block at once
            for T, g in zip(self.tables[pi], self.hashes[pi]):
                # (m, k) matrix of hash coordinates
                key_mat = g.batch(X_pi)

                # Insert into hashtable
                # Note: still one Python append per point, but the heavy math is vectorized.
                for obj_id, key_row in zip(ids_arr, key_mat):
                    key = tuple(key_row.tolist())
                    T[key].append(int(obj_id))

    # ---- query ----
    def search_and_solve(self, query, vector_store, dfunc):
        final_cands = []
        k = int(query['k'])
        data = query['count']
        if "text_query_embedding" in query:
            q = query['text_query_embedding']
        elif "vector" in query:
            q = query['vector']
        lists = [
            [f"{attr}:{value}" for value in values.keys()]
            for attr, values in data.items()
        ]

        # Cartesian product
        carte_query = list(itt.product(*lists))
        combo_info = {}

        search_time = 0.
        postprocessing_time = 0.

        start = time.time()
        for combo in carte_query:
            product = []
            for token in combo:
                attr, val = token.split(":", 1)
                product.append(self.proportions[token])
            requirement = math.ceil(math.prod(product) * k)

            # --- matching partitions containing all tokens in combo ---
            combo_set = set(combo)
            matching_parts = [
                p for p, tokens in self.partition_tokens
                if combo_set.issubset(tokens)
            ]

            combo_info[combo] = {
                "requirement": requirement,
                "partitions": matching_parts,
            }
        
        total_scan = 0
        for combo, info in combo_info.items():
            k_pi = info['requirement']
            all_cands = []
            for pi in info["partitions"]:   # for each matching partitions
                cands = []
                ell = len(self.tables[pi])
                k_star = info['requirement'] + math.ceil(2*ell/self.delta)
                for T, g in zip(self.tables[pi], self.hashes[pi]):
                    cands.extend(T.get(g(q), ()))
                cands = list(set(cands))[:k_star]
                all_cands.extend(cands)

                total_scan += len(cands)
            
            all_cands = list(set(all_cands))
            all_dists = [dfunc(q, vector_store[i]) for i in all_cands]
            pairs = sorted(zip(all_cands, all_dists), key=lambda x: x[1])

            k_pi_cands = [(obj_id, dist) for obj_id, dist in pairs[:k_pi]]
            final_cands.extend(k_pi_cands)
        
        search_time += time.time() - start

        final_cands = [(self.metadata_store[cand[0]], cand[1]) for cand in final_cands]

        if not len(final_cands):
            return None

        start = time.time()
        solver = build_solver(self.args)
        results = solver.solve(final_cands, query)
        postprocessing_time += time.time() - start

        results['search_time'] = search_time
        results['postprocessing_time'] = postprocessing_time
        results['total_scanned'] = total_scan

        return results




# class L2LSHJoint:
#     """
#     FAISS-based replacement for L2LSHCartesian.

#     - Same partitioning logic (Cartesian over protected attributes).
#     - One FAISS index per partition.
#     - Same search_and_solve() signature so the rest of the pipeline stays intact.
#     """
#     def __init__(self, args, d, metadata_store, protected_attrs):
#         self.args = args
#         self.d = d
#         self.metadata_store = metadata_store
#         self.c = float(args.c)
#         self.r = float(args.r)
#         self.w = float(args.w)
#         self.delta = float(args.delta)
#         self.rng = np.random.default_rng(args.seed)

#         # ------------- build Cartesian partitions (same as L2LSHCartesian) -------------
#         groups = defaultdict(list)
#         for a in protected_attrs:
#             k, v = a.split(":", 1)
#             groups[k].append(a)
#         carte_attrs = list(itt.product(*groups.values()))
#         self.all_carte_attrs = ["__".join(sorted(tup)) for tup in carte_attrs]

#         self.partitions = self.group_ids_by_partition()
#         self.partition_tokens = [
#             (p, set(p.split("__")))
#             for p in self.partitions
#         ]

#         self.proportions = {
#             attr: len([m for m in metadata_store if attr in m]) / len(metadata_store)
#             for attr in protected_attrs
#         }

#         # one FAISS index per partition
#         self.faiss_indexes = {}   # pi -> (faiss_index, np.ndarray[ids])
#         print(f"Built {len(self.partitions)} non-empty partitions.")

#     # -------- helpers copied from L2LSHCartesian --------
#     @staticmethod
#     def parse_kv_string(s):
#         """
#         Parses strings like 'age:30-39__gender:female__race:indian' (or with id)
#         into (id, features_dict). `id` is None if not present.
#         """
#         parts = s.split("__")
#         kv = {}
#         sid = None
#         for p in parts:
#             k, v = p.split(":", 1)
#             k = k.strip().lower()
#             v = v.strip().lower()
#             if k == "id":
#                 sid = int(v)
#             else:
#                 kv[k] = v
#         return sid, kv

#     @staticmethod
#     def feature_key(feat_dict):
#         """Order-agnostic canonical key for a feature dict (excluding id)."""
#         return tuple(sorted(feat_dict.items()))

#     def group_ids_by_partition(self):
#         """
#         partitions: list[str] like 'age:30-39__gender:female__race:indian'
#         datapoints: list[str] like 'id:0__gender:male__age:50-59__race:east asian'
#         returns: dict[str, list[int]] mapping the *original partition string* to ids
#         """
#         partitions = self.all_carte_attrs
#         datapoints = self.metadata_store

#         features_to_partition = {}
#         out = {p: [] for p in partitions}
#         for p in partitions:
#             _, feats = self.parse_kv_string(p)
#             features_to_partition[self.feature_key(feats)] = p

#         for d in datapoints:
#             sid, feats = self.parse_kv_string(d)
#             if sid is None:
#                 continue
#             key = self.feature_key(feats)
#             part = features_to_partition.get(key)
#             if part is not None:
#                 out[part].append(sid)

#         out = {k: v for k, v in out.items() if len(v)}
#         return out

#     # -------- FAISS indexing --------
#     def build_index(self, X: np.ndarray):
#         """
#         X: (n, d) array of float32. Row i corresponds to object id i.
#         Builds a FAISS index per partition.
#         """
#         assert X.shape[1] == self.d, "Dimension mismatch in build_index"
#         X = X.astype('float32')

#         for pi, ids in self.partitions.items():
#             if not ids:
#                 continue
#             ids_arr = np.asarray(ids, dtype=np.int64)
#             X_pi = X[ids_arr]  # (m, d)

#             # simplest FAISS index: exact L2 (can swap to IVF if needed)
#             index = faiss.IndexFlatL2(self.d)
#             index.add(X_pi)

#             self.faiss_indexes[pi] = (index, ids_arr)
#             # print(f"Built FAISS index for partition {pi} with {len(ids_arr)} points.")

#     # -------- query --------
#     def search_and_solve(self, query, vector_store, dfunc):
#         """
#         query: dict with
#             - 'text_query_embedding' : np.ndarray(d,)
#             - 'count' : nested dict of desired counts per attribute value

#         vector_store: not used here (we rely on FAISS distances),
#                       but kept for API compatibility.
#         dfunc:        not used, can be a dummy lambda.
#         """
#         final_cands = []
#         data = query['count']
#         k = int(query['k'])
#         if "query_vector" in query:
#             q = query['query_vector'].astype('float32')
#         else:
#             q = query['text_query_embedding'].astype('float32')
#         lists = [
#             [f"{attr}:{value}" for value in values.keys()]
#             for attr, values in data.items()
#         ]

#         carte_query = list(itt.product(*lists))
#         combo_info = {}

#         search_time = 0.0
#         postprocessing_time = 0.0

#         # ----- precompute requirements per Cartesian combo -----
#         start = time.time()
#         for combo in carte_query:
#             product = []
#             for token in combo:
#                 attr, val = token.split(":", 1)
#                 product.append(self.proportions[token])
#             requirement = math.ceil(math.prod(product) * k)

#             # --- matching partitions containing all tokens in combo ---
#             combo_set = set(combo)
#             matching_parts = [
#                 p for p, tokens in self.partition_tokens
#                 if combo_set.issubset(tokens)
#             ]

#             combo_info[combo] = {
#                 "requirement": requirement,
#                 "partitions": matching_parts,
#             }
        
#         total_scan = 0

#         # ----- retrieval per combo using FAISS -----
#         for combo, info in combo_info.items():
#             k_pi = info['requirement']
#             if k_pi <= 0:
#                 continue

#             all_cands = []
#             all_dists = []

#             for pi in info["partitions"]:
#                 if pi not in self.faiss_indexes:
#                     continue
#                 index, ids_arr = self.faiss_indexes[pi]
#                 ell = 1  # not LSH tables anymore; keep for compatibility
#                 # oversample a bit, but not beyond partition size
#                 k_star = min(len(ids_arr), k_pi + math.ceil(2 * ell / self.delta))

#                 D, I = index.search(q.reshape(1, -1), k_star)  # shapes (1, k_star)
#                 D = D[0]
#                 I = I[0]

#                 for local_idx, dist in zip(I, D):
#                     if local_idx < 0:
#                         continue
#                     obj_id = int(ids_arr[local_idx])
#                     all_cands.append(obj_id)
#                     all_dists.append(float(dist))

#                 total_scan += k_star

#             # de-duplicate and sort
#             if all_cands:
#                 # group by obj_id, keep smallest distance
#                 best_dist = {}
#                 for obj_id, dist in zip(all_cands, all_dists):
#                     if obj_id not in best_dist or dist < best_dist[obj_id]:
#                         best_dist[obj_id] = dist

#                 pairs = sorted(best_dist.items(), key=lambda x: x[1])
#                 k_pi_cands = pairs[:k_pi]
#                 final_cands.extend(k_pi_cands)

#         search_time += time.time() - start

#         # map from id -> metadata string, as in original code
#         final_cands = [(self.metadata_store[cand_id], dist) for cand_id, dist in final_cands]

#         if not len(final_cands):
#             return None, None

#         start = time.time()
#         solver = build_solver(self.args)
#         results = solver.solve(final_cands, query)
#         postprocessing_time += time.time() - start

#         results['search_time'] = search_time
#         results['postprocessing_time'] = postprocessing_time

#         return results, total_scan