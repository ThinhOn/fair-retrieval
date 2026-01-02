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
class HyperplaneHash:
    """
    Random hyperplane LSH for angular / cosine similarity.

    h(x) = 1 if a·x >= 0 else 0

    NOTE: For proper angular LSH, x should be L2-normalized before hashing.
    """
    a: np.ndarray  # shape (d,)

    @staticmethod
    def sample(d: int, rng: np.random.Generator) -> "HyperplaneHash":
        # Gaussian normal; only the sign matters
        a = rng.normal(size=d)
        return HyperplaneHash(a=a)

    def __call__(self, x: np.ndarray) -> int:
        # Bit valued hash (0 or 1)
        return int(self.a @ x >= 0.0)

# ------------------------------------------------------
# Compound hash g(x) = (h1(x), ..., hk(x)) for one table
# ------------------------------------------------------

@dataclass
class CompoundHash:
    funcs: Tuple[Callable[[np.ndarray], int], ...]  # length k
    A: Optional[np.ndarray] = None   # (k, d) for L2

    def __post_init__(self):
        if not self.funcs:
            return

        first = self.funcs[0]
        if isinstance(first, HyperplaneHash):
            As = []
            for f in self.funcs:
                assert isinstance(f, HyperplaneHash), "Mixed hash types not supported in hyperplane vectorized path"
                As.append(f.a)
            self.A = np.stack(As, axis=0)      # (k, d)
            return

    def __call__(self, x: np.ndarray) -> HashKey:
        """
        Hash a single vector x. Keeps old API: returns a tuple[int,...].
        Uses vectorized path if available.
        """
        if self.A is not None:
            # (k,d) @ (d,) -> (k,)
            proj = self.A @ x
            bits = (proj >= 0).astype(np.int64)
            return tuple(bits.tolist())
        return tuple(f(x) for f in self.funcs)

    def batch(self, X: np.ndarray) -> np.ndarray:
        """
        Hash a batch of vectors.
        X: (n, d)
        Returns: (n, k) int64 array of hash coordinates.
        """
        if self.A is not None:
            proj = X @ self.A.T  # (n,k)
            bits = (proj >= 0).astype(np.int64)
            return bits
            
def make_compound_family(
    family_sampler: Callable[[], Callable[[np.ndarray], int]],
    mu: int,
    rng: np.random.Generator,
) -> CompoundHash:
    funcs = tuple(family_sampler() for _ in range(mu))
    return CompoundHash(funcs)


class AngularLSHSingle:
    r"""
    Generic LSH index with $\ell$ tables; each table uses a compound hash of k base hashes.
    Buckets store integer ids; you can store payloads separately if desired.
    """
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

        self.protected_attrs = protected_attrs

        self.partitions = self.group_ids_by_partition()

        self.tables = {}
        self.hashes = {}
        # Interpret args.r as angular threshold (radians).
        # For "far" points, use angle c * r, clipped to π.
        theta1 = self.r
        theta2 = min(self.c * self.r, math.pi)

        p1 = self.collision_probability(theta1, self.w)
        p2 = self.collision_probability(theta2, self.w)
        for pi, oids in self.partitions.items():    # for each partition \pi
            if not oids:
                continue

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
                def sampler() -> Callable[[np.ndarray], int]:
                    h = HyperplaneHash.sample(d=self.d, rng=self.rng)
                    return h
                self.hashes[pi].append(make_compound_family(sampler, mu=mu, rng=self.rng))


    @staticmethod
    def collision_probability(theta: float, w: Optional[float] = None) -> float:
        """
        Collision probability for random hyperplane LSH between two vectors
        at angle theta:
            p(theta) = 1 - theta / pi
        The `w` parameter is ignored but kept for API compatibility.
        """
        theta = float(theta)
        theta = max(0.0, min(theta, math.pi))
        return 1.0 - theta / math.pi
    

    def group_ids_by_partition(self):
        partitions = self.protected_attrs
        datapoints = self.metadata_store
        out = {p: [] for p in partitions}  # ensure all partitions exist with empty lists
        for meta in datapoints:
            meta = meta.split('__')
            oid = int(meta[0].split(':')[1])
            for part in meta[1:]:
                out[part].append(oid)
        return out


    def build_index(self, X: np.ndarray):
        """
        NEW: fast bulk indexing.
        X: (n, d), where row i corresponds to object id i (as used in metadata_store).
        Uses precomputed partitions and vectorized hashing.
        """
        assert X.shape[1] == self.d, "Dimension mismatch in build_index"

        # For each partition pi, we know exactly which ids belong to it.
        for pi, ids in self.partitions.items():
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
        beta_hat = query['count']
        if "text_query_embedding" in query:
            q = query['text_query_embedding']
        elif "vector" in query:
            q = query['vector']
        queried_parts = [
            [f"{attr}:{value}" for value in values.keys()]
            for attr, values in beta_hat.items()
        ]
        queried_parts = [item for sublist in queried_parts for item in sublist] # flatten
        # print(queried_parts)
        # exit()
        
        final_cands = []
        total_scan = 0

        search_time = 0.
        postprocessing_time = 0.

        start = time.time()
        for part in queried_parts:
            cands = []
            attr, val = part.split(":", 1)
            k_pi = beta_hat[attr][val]
            ell = len(self.tables[part])
            k_star = k_pi + math.ceil(2*ell/self.delta)

            for T, g in zip(self.tables[part], self.hashes[part]):
                cands.extend(T.get(g(q), ()))

            cands = list(set(cands))[:k_star]
            total_scan += len(cands)

            dists = [dfunc(q, vector_store[i]) for i in cands]
            pairs = sorted(zip(cands, dists), key=lambda x: x[1])

            k_pi_cands = [(obj_id, dist) for obj_id, dist in pairs[:k_pi]]
            final_cands.extend(k_pi_cands)

        search_time += time.time() - start
        
        final_cands = [(self.metadata_store[cand[0]], cand[1]) for cand in final_cands]

        start = time.time()
        solver = build_solver(self.args)
        results = solver.solve(final_cands, query)
        postprocessing_time += time.time() - start

        results['search_time'] = search_time
        results['postprocessing_time'] = postprocessing_time
        results['total_scanned'] = total_scan

        return results









# class L2LSHSingle:
#     r"""
#     FAISS-based replacement for L2LSHSingle.

#     - Same partitioning over single protected attributes.
#     - Same query format: query["count"][attr][val] = k_pi.
#     - One FAISS index per partition (attribute value).
#     - ℓ per partition is still computed from the LSH theory and
#       used only as an *oversampling* knob to decide k_star.
#     """
#     def __init__(
#         self,
#         args,
#         d,
#         metadata_store,
#         protected_attrs,
#     ):
#         self.args = args
#         self.d = d
#         self.metadata_store = metadata_store
#         self.protected_attrs = protected_attrs

#         self.c = float(args.c)
#         self.r = float(args.r)
#         self.w = float(args.w)
#         self.delta = float(args.delta)
#         self.K = int(args.max_K)
#         self.rng = np.random.default_rng(args.seed)

#         # same partitioning as L2LSHSingle: one partition per protected attribute value
#         self.partitions = self.group_ids_by_partition()

#         self.tables = {}
#         self.hashes = {}

#         p1 = self.collision_probability(self.r, self.w)
#         p2 = self.collision_probability(self.c * self.r, self.w)
#         for pi, oids in self.partitions.items():    # for each partition \pi
#             mu = math.ceil( math.log(len(oids)) / math.log(1/p2) )
#             # mu = args.mu
#             mu = max(1, mu)
#             ell = math.ceil( math.log(self.delta) / math.log(1 - p1**mu) )
#             # ell = args.ell
#             self.tables[pi] = [defaultdict(list) for _ in range(ell)]
#             self.hashes[pi] = []

#             # print(f"Partition {pi}: K={mu}, L={ell}")

#             for _ in range(ell):
#                 # if family == "l2":
#                 def sampler() -> Callable[[np.ndarray], int]:
#                     h = L2Hash.sample(d=self.d, w=self.w, rng=self.rng)
#                     return h
#                 # else:
#                 #     def sampler() -> Callable[[np.ndarray], int]:
#                 #         h = HyperplaneHash.sample(d=self.d, rng=self.rng)
#                 #         return h
#                 self.hashes[pi].append(make_compound_family(sampler, mu=mu, rng=self.rng))
#             # print(mu, ell)
#             # print(self.hashes)
#             # exit()

#     @staticmethod
#     def collision_probability(r, w):
#         """
#         Compute collision probability for L2 LSH given distance r and bucket width w.
#         Kept so we can reuse the same ℓ formula as an oversampling knob.
#         """
#         integrand = lambda t: (1 - t / w) * (1 / r) * norm.pdf(t / r)
#         result, _ = quad(integrand, 0, w)
#         return 2 * result

#     def group_ids_by_partition(self):
#         """
#         partitions = self.protected_attrs  (strings like 'gender:male', 'race:asian', ...)
#         metadata_store = list of strings like 'id:0__gender:male__age:50-59__race:east asian'

#         Returns:
#             dict[part] -> list of ids that have that attribute value.
#         """
#         partitions = self.protected_attrs
#         datapoints = self.metadata_store
#         out = {p: [] for p in partitions}
#         for meta in datapoints:
#             fields = meta.split('__')
#             oid = int(fields[0].split(':')[1])
#             for part in fields[1:]:
#                 if part in out:
#                     out[part].append(oid)
#         return out

#     # ---- indexing with FAISS ----

#     def build_index(self, X: np.ndarray):
#         """
#         X: (n, d) float32 array, where row i corresponds to object id i.
#         Builds a FAISS IndexFlatL2 per partition.
#         """
#         assert X.shape[1] == self.d, "Dimension mismatch in build_index"
#         X = X.astype("float32")

#         for part, ids in self.partitions.items():
#             if not ids:
#                 continue
#             ids_arr = np.asarray(ids, dtype=np.int64)
#             X_pi = X[ids_arr]  # subset (m, d)

#             index = faiss.IndexFlatL2(self.d)
#             index.add(X_pi)

#             self.faiss_indexes[part] = (index, ids_arr)
#             ell = self.ell_per_partition.get(part, 1)
#             print(f"[FaissSingleIndex] partition {part}: {len(ids_arr)} points, ell={ell}")

#     # ---- query ----

#     def search_and_solve(self, query, vector_store, dfunc):
#         """
#         query: dict with
#             - 'count': nested dict, beta_hat[attr][val] = k_pi
#             - 'text_query_embedding': np.ndarray(d,)

#         dfunc: ignored (FAISS provides distances), kept for API compatibility.

#         Returns:
#             results, total_scan
#         """
#         beta_hat = query["count"]
#         if "text_query_embedding" in query:
#             q = query["text_query_embedding"].astype("float32")
#         else:
#             q = query["query_vector"].astype("float32")

#         # flatten attribute:value tokens from the query, like original code:
#         # queried_parts = ["gender:male", "gender:female", "age:young", ...]
#         queried_parts = [
#             [f"{attr}:{value}" for value in values.keys()]
#             for attr, values in beta_hat.items()
#         ]
#         queried_parts = [item for sublist in queried_parts for item in sublist]

#         final_cands = []
#         total_scan = 0

#         search_time = 0.0
#         postprocessing_time = 0.0

#         start = time.time()
#         for part in queried_parts:
#             attr, val = part.split(":", 1)
#             k_pi = beta_hat[attr][val]
#             if k_pi <= 0:
#                 continue

#             if part not in self.faiss_indexes:
#                 continue

#             index, ids_arr = self.faiss_indexes[part]
#             ell = self.ell_per_partition.get(part, 1)

#             # oversample by ℓ, as in original: k_star = k_pi + ceil(2*ell/delta)
#             k_star = k_pi + math.ceil(2 * ell / self.delta)
#             k_star = min(k_star, len(ids_arr))
#             if k_star <= 0:
#                 continue

#             # FAISS search
#             D, I = index.search(q.reshape(1, -1), k_star)  # shapes (1, k_star)
#             D = D[0]
#             I = I[0]

#             # map local indices to global object ids
#             pairs = []
#             for local_idx, dist in zip(I, D):
#                 if local_idx < 0:
#                     continue
#                 obj_id = int(ids_arr[local_idx])
#                 pairs.append((obj_id, float(dist)))

#             total_scan += len(pairs)

#             # sort and keep top k_pi
#             pairs = sorted(pairs, key=lambda x: x[1])
#             k_pi_cands = pairs[:k_pi]
#             final_cands.extend(k_pi_cands)

#         search_time += time.time() - start

#         # map id -> metadata string, same as original
#         final_cands = [(self.metadata_store[cand_id], dist) for cand_id, dist in final_cands]

#         # no candidates at all — return sentinel
#         if not final_cands:
#             return None, total_scan

#         # fairness solver unchanged
#         start = time.time()
#         solver = build_solver(self.args)
#         results = solver.solve(final_cands, query)
#         postprocessing_time += time.time() - start

#         results["search_time"] = search_time
#         results["postprocessing_time"] = postprocessing_time

#         return results, total_scan
