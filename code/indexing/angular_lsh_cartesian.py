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



class AngularLSHCartesian:
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
        # self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.c = float(args.c)
        self.r = float(args.r)
        self.w = float(args.w)
        self.delta = float(args.delta)
        # K = args.max_K
        self.rng = np.random.default_rng(args.seed)

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


    def build_index(self, X: np.ndarray):
        """
        Fast bulk indexing.
        X: (n, d), where row i corresponds to object id i (as used in metadata_store).
        Uses precomputed partitions and vectorized hashing + grouping.
        """
        assert X.shape[1] == self.d, "Dimension mismatch in build_index"

        for pi, ids in self.partitions.items():
            # print(f"Indexing partition {pi} with {len(ids)} points, ell = {len(self.tables[pi])}...")
            if not ids:
                continue

            ids_arr = np.asarray(ids, dtype=np.int64)
            X_pi = X[ids_arr]  # (m, d)

            for T, g in zip(self.tables[pi], self.hashes[pi]):
                # (m, k) int64 matrix of hash coordinates
                key_mat = g.batch(X_pi)  # shape (m, k)

                # Group by unique hash key to avoid per-vector Python overhead
                # keys_unique: (u, k), inv: (m,) giving the group index for each row
                keys_unique, inv = np.unique(key_mat, axis=0, return_inverse=True)

                # For each unique key, take all ids that map to it
                for bucket_idx, key_vec in enumerate(keys_unique):
                    bucket_ids = ids_arr[inv == bucket_idx]
                    if bucket_ids.size == 0:
                        continue
                    key = tuple(int(v) for v in key_vec)
                    # extend once per bucket instead of once per vector
                    T[key].extend(map(int, bucket_ids))

    # ---- query ----
    def search_and_solve(self, query, vector_store, dfunc):
        final_cands = []
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
            counts = []
            for token in combo:
                attr, val = token.split(":", 1)
                counts.append(data[attr][val])
            requirement = min(counts)

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
                    # print(f"hash: {g(q)}")
                    # print(f"table of hash: {T.get(g(q), None)}")
                    # exit()
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