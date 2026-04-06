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

# ── billion-scale helpers ──────────────────────────────────────────────────
ATTR_NAMES  = ["A1", "A2", "A3"]
ATTR_VALUES = [
    ["V11", "V12", "V13"],
    ["V21", "V22", "V23", "V24"],
    ["V31", "V32", "V33"],
]

def reconstruct_metadata(oid: int, attr_codes: np.ndarray) -> str:
    """Reconstruct metadata string from integer attribute codes.
    attr_codes: 1-D array of length m with values matching ATTR_VALUES indices.
    Returns e.g. 'id:0__A1:V12__A2:V23__A3:V31'
    """
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

class L2LSHCartesian:
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

        # if family == "l2":
        def sampler() -> Callable[[np.ndarray], int]:
            h = L2Hash.sample(d=self.d, w=self.w, rng=self.rng)
            return h
        # else:
        #     def sampler() -> Callable[[np.ndarray], int]:
        #         h = HyperplaneHash.sample(d=self.d, rng=self.rng)
        #         return h

        # p1 = self.collision_probability(self.r, self.w)
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

    # def find_R_for_p1(self, target_p1, w, R_min=1e-9, R_max=1e4, tol=1e-6, max_iter=60):
    #     """
    #     Solve for R such that collision_probability(R, w) = target_p1.
        
    #     Parameters:
    #         target_p1 : float   # desired collision probability for "near" points
    #         w         : float   # bucket width for L2-LSH
    #         R_min     : float   # lower search bound
    #         R_max     : float   # upper search bound
    #         tol       : float   # absolute tolerance on p1
    #         max_iter  : int     # max binary search steps
        
    #     Returns:
    #         float: value of R such that p1 ≈ target_p1
    #     """
    #     # Safety checks
    #     if not (0 < target_p1 < 1):
    #         raise ValueError("target_p1 must be between 0 and 1")

    #     # Expand R_max until the collision probability drops below target_p1
    #     while self.collision_probability(R_max, w) > target_p1:
    #         R_max *= 2
    #         if R_max > 1e12:
    #             raise RuntimeError("R too large; check w or target_p1 settings.")

    #     # Binary search
    #     for _ in range(max_iter):
    #         R_mid = 0.5 * (R_min + R_max)
    #         p_mid = self.collision_probability(R_mid, w)

    #         if abs(p_mid - target_p1) < tol:
    #             return R_mid

    #         if p_mid > target_p1:
    #             R_min = R_mid  # need larger R to reduce collision probability
    #         else:
    #             R_max = R_mid

    #     return 0.5 * (R_min + R_max)  # best estimate

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
        - Billion-scale: reads integer codes directly from attrs_memmap (fast, O(n)).
        - Small datasets: parses metadata strings as before.
        Returns dict[partition_str, np.ndarray[int64]].
        """
        partitions = self.all_carte_attrs

        if self.attrs_memmap is not None:
            # ── Billion-scale path ────────────────────────────────────────────
            code_to_partition = {}
            for p in partitions:
                tokens = p.split("__")
                codes = []
                valid = True
                for token in tokens:
                    attr, val = token.split(":", 1)
                    if attr in ATTR_NAMES:
                        j = ATTR_NAMES.index(attr)
                        if val in ATTR_VALUES[j]:
                            codes.append(ATTR_VALUES[j].index(val))
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                if valid:
                    code_to_partition[tuple(codes)] = p

            out = {p: [] for p in partitions}
            n = self.attrs_memmap.shape[0]
            chunk = 1_000_000
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                batch = self.attrs_memmap[start:end]   # (chunk, m) uint8
                for local_i in range(end - start):
                    key = tuple(int(c) for c in batch[local_i])
                    p = code_to_partition.get(key)
                    if p is not None:
                        out[p].append(start + local_i)
            # convert to numpy arrays for memory efficiency
            out = {k: np.asarray(v, dtype=np.int64)
                   for k, v in out.items() if len(v)}
            return out

        else:
            # ── Small-dataset path (original) ─────────────────────────────────
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


    def build_index(self, X: np.ndarray, chunk_size: int = 500_000):
        """
        Bulk indexing, one partition at a time in chunks.

        For billion-scale datasets (attrs_memmap is set), X is a memmap and
        ids_arr can be very large. We process each partition in chunks of
        chunk_size rows so that only ~chunk_size * d * 4 bytes live in RAM
        at once (e.g. 500k * 128 * 4 = 256 MB per chunk).

        For small datasets the behaviour is identical to before.
        """
        assert X.shape[1] == self.d, "Dimension mismatch in build_index"

        for pi, ids in self.partitions.items():
            if not len(ids):
                print(f"Warning: partition '{pi}' has no ids; skipping indexing for this partition.")
                continue

            ids_arr = np.asarray(ids, dtype=np.int64)
            n_pi = len(ids_arr)

            for T, g in zip(self.tables[pi], self.hashes[pi]):
                # process this partition in chunks to bound RAM usage
                for chunk_start in range(0, n_pi, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_pi)
                    ids_chunk = ids_arr[chunk_start:chunk_end]
                    X_chunk = X[ids_chunk]   # memmap random access — loads only this slice

                    # (chunk, mu) int64 hash keys
                    key_mat = g.batch(X_chunk)

                    keys_unique, inv = np.unique(key_mat, axis=0, return_inverse=True)
                    for bucket_idx, key_vec in enumerate(keys_unique):
                        bucket_ids = ids_chunk[inv == bucket_idx]
                        if bucket_ids.size == 0:
                            continue
                        key = tuple(int(v) for v in key_vec)
                        T[key].extend(map(int, bucket_ids))

    # ---- query ----
    def search_and_solve(self, query, vector_store, dfunc, c_min=None):
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
                constraint = data[attr][val]
                # ranged constraint: (lb, ub) — use ub as the per-partition ceiling
                # exact constraint: integer — use as-is
                ub = constraint[1] if isinstance(constraint, (tuple, list)) else constraint
                counts.append(ub)
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

        n_combos = len(combo_info)
        total_scan = 0
        for combo, info in combo_info.items():
            k_pi = info['requirement']

            # scale up k_pi so that the total pool reaches c_min
            if c_min is not None:
                k_pi = max(k_pi, math.ceil(c_min / n_combos))

            all_cands = []
            for pi in info["partitions"]:   # for each matching partition
                cands = []
                ell = len(self.tables[pi])
                k_star = k_pi + math.ceil(2*ell/self.delta)
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

        # map ids to metadata strings
        if self.attrs_memmap is not None:
            # billion-scale: reconstruct string on the fly from integer codes
            final_cands = [
                (reconstruct_metadata(int(cand[0]), self.attrs_memmap[int(cand[0])]), cand[1])
                for cand in final_cands
            ]
        else:
            final_cands = [(self.metadata_store[cand[0]], cand[1]) for cand in final_cands]

        # deduplicate by id, preserving distance order
        seen = set()
        deduped = []
        for meta, dist in sorted(final_cands, key=lambda x: x[1]):
            oid = meta.split("__")[0]
            if oid not in seen:
                seen.add(oid)
                deduped.append((meta, dist))
        final_cands = deduped

        # ── fallback: pad with random points from matching partitions ─────────
        # If LSH buckets are exhausted but |C| < c_min, sample uniformly from
        # the matching partition id lists (already in memory) until c_min is
        # reached or no more candidates are available.
        if c_min is not None and len(final_cands) < c_min:
            selected_ids = {meta.split("__")[0] for meta, _ in final_cands}

            # collect all unselected ids from matching partitions
            pool_ids = []
            for info in combo_info.values():
                for pi in info["partitions"]:
                    for oid in self.partitions[pi]:
                        sid = f"id:{oid}"
                        if sid not in selected_ids:
                            pool_ids.append(oid)

            pool_ids = list(set(pool_ids))
            rng = np.random.default_rng()
            rng.shuffle(pool_ids)

            needed = c_min - len(final_cands)
            for oid in pool_ids[:needed]:
                if self.attrs_memmap is not None:
                    meta = reconstruct_metadata(int(oid), self.attrs_memmap[int(oid)])
                else:
                    meta = self.metadata_store[oid]
                dist = dfunc(q, vector_store[oid])
                final_cands.append((meta, dist))
                selected_ids.add(f"id:{oid}")
        # ─────────────────────────────────────────────────────────────────────

        if not len(final_cands):
            return None

        start = time.time()
        if self.args.m > 1:
            solver = build_solver(self.args)
            results = solver.solve(final_cands, query)
        else:
            count = {k: {v: 0 for v in reqs} for k, reqs in query['count'].items()}
            for text, _ in final_cands:
                parts = text.split("__")
                for part in parts:
                    if ":" not in part:
                        continue
                    key, val = part.split(":")
                    if key in query['count'] and val in query['count'][key]:
                        count[key][val] += 1
            results = {
                'objective': sum(cand[1] for cand in final_cands),
                'selected': [cand[0] for cand in final_cands],
                'count': count,
            }

        postprocessing_time += time.time() - start

        results['search_time'] = search_time
        results['postprocessing_time'] = postprocessing_time
        results['total_scanned'] = total_scan
        results['pool_size'] = len(final_cands)

        return results