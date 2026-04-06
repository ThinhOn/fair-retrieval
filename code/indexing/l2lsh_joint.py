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

# ── billion-scale schema (must match generate_billion.py) ──────────────────
ATTR_NAMES  = ["A1", "A2", "A3"]
ATTR_VALUES = [
    ["V11", "V12", "V13"],
    ["V21", "V22", "V23", "V24"],
    ["V31", "V32", "V33"],
]

def reconstruct_metadata(oid: int, attr_codes: np.ndarray) -> str:
    """Reconstruct metadata string from integer attribute codes."""
    parts = [f"id:{oid}"]
    for j, code in enumerate(attr_codes):
        parts.append(f"{ATTR_NAMES[j]}:{ATTR_VALUES[j][int(code)]}")
    return "__".join(parts)

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
        attrs_memmap=None,
    ):
        self.args = args
        self.d = d
        # attrs_memmap: (n, m) uint8 memmap for billion-scale; None for small datasets
        self.attrs_memmap = attrs_memmap
        # metadata_store: list[str] for small datasets; None for billion-scale
        self.metadata_store = metadata_store
        self.c = float(args.c)
        self.r = float(args.r)
        self.w = float(args.w)
        self.delta = float(args.delta)
        self.rng = np.random.default_rng(args.seed)

        if attrs_memmap is not None:
            # billion-scale: compute proportions from integer codes in one pass
            n = attrs_memmap.shape[0]
            token_counts = defaultdict(int)
            chunk = 1_000_000
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                batch = attrs_memmap[start:end]
                for local_i in range(end - start):
                    for j in range(batch.shape[1]):
                        token = f"{ATTR_NAMES[j]}:{ATTR_VALUES[j][int(batch[local_i, j])]}"
                        token_counts[token] += 1
            self.proportions = {attr: token_counts.get(attr, 0) / n
                                for attr in protected_attrs}
        else:
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
        Two paths:
        - Billion-scale: reads integer codes from attrs_memmap in chunks.
        - Small datasets: parses metadata strings as before.
        Returns dict[partition_str, list[int]].
        """
        partitions = self.all_carte_attrs

        if self.attrs_memmap is not None:
            # build lookup: tuple of integer codes -> partition string
            code_to_partition = {}
            for p in partitions:
                _, feats = self.parse_kv_string(p)
                codes = []
                valid = True
                for j, name in enumerate(ATTR_NAMES[:self.attrs_memmap.shape[1]]):
                    val_str = feats.get(name.lower(), feats.get(name, None))
                    if val_str is None:
                        valid = False
                        break
                    try:
                        codes.append(ATTR_VALUES[j].index(val_str))
                    except ValueError:
                        valid = False
                        break
                if valid:
                    code_to_partition[tuple(codes)] = p

            out = {p: [] for p in partitions}
            n = self.attrs_memmap.shape[0]
            chunk = 1_000_000
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                batch = self.attrs_memmap[start:end]
                for local_i in range(end - start):
                    key = tuple(int(c) for c in batch[local_i])
                    p = code_to_partition.get(key)
                    if p is not None:
                        out[p].append(start + local_i)
            out = {k: np.asarray(v, dtype=np.int64)
                   for k, v in out.items() if len(v)}
            return out

        else:
            # standard path
            datapoints = self.metadata_store
            features_to_partition = {}
            out = {p: [] for p in partitions}
            for p in partitions:
                _, feats = self.parse_kv_string(p)
                features_to_partition[self.feature_key(feats)] = p
            for d in datapoints:
                sid, feats = self.parse_kv_string(d)
                if sid is None:
                    continue
                key = self.feature_key(feats)
                part = features_to_partition.get(key)
                if part is not None:
                    out[part].append(sid)
            out = {k: v for k, v in out.items() if len(v)}
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

    def build_index(self, X: np.ndarray, chunk_size: int = 500_000):
        """
        Bulk indexing, one partition at a time in chunks.
        For billion-scale datasets, processes each partition in chunks of
        chunk_size rows so that only ~chunk_size * d * 4 bytes live in RAM.
        """
        assert X.shape[1] == self.d, "Dimension mismatch in build_index"

        for pi, ids in self.partitions.items():
            if not ids:
                continue
            ids_arr = np.asarray(ids, dtype=np.int64)
            n_pi = len(ids_arr)

            for T, g in zip(self.tables[pi], self.hashes[pi]):
                for chunk_start in range(0, n_pi, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_pi)
                    ids_chunk = ids_arr[chunk_start:chunk_end]
                    X_chunk = X[ids_chunk]   # memmap random access

                    key_mat = g.batch(X_chunk)

                    for obj_id, key_row in zip(ids_chunk, key_mat):
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

        if self.attrs_memmap is not None:
            final_cands = [
                (reconstruct_metadata(int(cand[0]), self.attrs_memmap[int(cand[0])]), cand[1])
                for cand in final_cands
            ]
        else:
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