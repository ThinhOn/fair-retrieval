from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import math
import time
import numpy as np
import tqdm
from dataclasses import dataclass
from typing import Tuple, Dict, List, Iterable, Optional, Callable, Any
from collections import defaultdict
import itertools as itt
from scipy.stats import norm
from scipy.integrate import quad
from solver import build_solver
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import shelve
import sqlite3


# ── on-disk per-partition bucket store ───────────────────────────────────────
# Backend = SQLite, NOT shelve. On Windows shelve falls back to dbm.dumb, which
# rewrites its whole index on every key (O(K^2) + an fsync per key): ~31 s to
# write 5600 keys. SQLite writes the same in ~0.02 s (one batched transaction)
# and still gives fast random-access point reads by key — essential at billion
# scale. Each partition is one .sqlite file; values are int64 id arrays as BLOBs.

def _sql_path(cache_path: str) -> str:
    return cache_path + ".sqlite"


def _sql_exists(cache_path: str) -> bool:
    return os.path.exists(_sql_path(cache_path))


def _sql_write(cache_path: str, acc: list) -> None:
    """Write acc[t_idx][kb] -> [id arrays] to a fresh SQLite file, one txn."""
    p = _sql_path(cache_path)
    if os.path.exists(p):
        os.remove(p)
    con = sqlite3.connect(p)
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("CREATE TABLE kv (k TEXT PRIMARY KEY, v BLOB)")

        def rows():
            for t_idx in range(len(acc)):
                for kb, arrs in acc[t_idx].items():
                    # ids < n <= 1e9 < 2**31, so int32 suffices — halves the
                    # index size on disk and the accumulation RAM.
                    ids = np.concatenate(arrs).astype(np.int32)
                    yield (f"{t_idx}:{kb.hex()}", ids.tobytes())

        con.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)", rows())
        con.commit()
    finally:
        con.close()


def _sql_open_ro(cache_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{_sql_path(cache_path)}?mode=ro", uri=True)


def _sql_get(con: sqlite3.Connection, t_idx: int, kb: bytes) -> list:
    row = con.execute("SELECT v FROM kv WHERE k=?",
                      (f"{t_idx}:{kb.hex()}",)).fetchone()
    if row is None:
        return []
    return np.frombuffer(row[0], dtype=np.int32).tolist()

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
            self.A = np.stack(As, axis=0).astype(np.float32)
            self.b = np.asarray(bs, dtype=np.float32)
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



# ── helpers for fast parallel index build ────────────────────────────────────

def _group_key_rows(key_mat: np.ndarray):
    """
    Group identical hash-key rows of a (n, mu) int64 matrix.

    Returns (unique_keys, inverse) where:
      - unique_keys[k] is a stable `bytes` key for the k-th unique row,
      - inverse[i] is the index of row i's unique key.

    Using the raw row bytes as the key (instead of a polynomial int encoding)
    is overflow-free for ANY concatenation length mu. The old int encoding
    (base 1_000_003 ** i) overflowed int64 once mu > 3, which made the
    theory-prescribed auto-scaled mu (mu=0 → mu≈log n / log(1/p2), often 8–13
    on large partitions) impossible to build.
    """
    uniq, inverse = np.unique(key_mat, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    uniq = np.ascontiguousarray(uniq)
    keys = [uniq[k].tobytes() for k in range(uniq.shape[0])]
    return keys, inverse


def _query_key(g, q) -> bytes:
    """Stable bytes key for a single query vector under compound hash g."""
    return np.asarray(g(q), dtype=np.int64).tobytes()


def _index_partition_worker(pi, ids, hashes, vec_path, vec_shape,
                            chunk_size, cache_path=None):
    """
    Build LSH tables for one partition.

    When cache_path is given, each partition's buckets are accumulated in RAM
    across all chunks and then written to shelve ONCE per key. RAM is bounded
    to O(n_pi * ell) ids (plus one chunk of vectors). This avoids the
    quadratic read-modify-write (`db[k] = db[k] + ids` per chunk) that makes
    dbm.dumb — the only shelve backend on Windows — pathologically slow when
    coarse (small-mu) buckets hold millions of ids each.

    Optimisations:
    1. Chunk-outer / tables-inner: each chunk read from memmap ONCE for all
       ell tables (ell× fewer disk reads).
    2. np.unique grouping: replaces per-row Python append loop.
    3. One shelve write per key (no per-chunk read-modify-write).
    """
    import numpy as np
    from collections import defaultdict as _dd

    X        = np.memmap(vec_path, dtype=np.float32, mode="r", shape=vec_shape)
    ids_arr  = np.sort(np.asarray(ids, dtype=np.int64))
    n_pi     = len(ids_arr)
    n_tables = len(hashes)

    PRINT_FREQ = 10

    if cache_path is not None:
        # ── Accumulate in RAM (bucket -> list of id arrays), write once ───────
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        acc = [_dd(list) for _ in range(n_tables)]   # acc[t][kb] -> [id arrays]
        for chunk_start in range(0, n_pi, chunk_size):
            if chunk_start % (chunk_size * PRINT_FREQ) == 0:
                print(f"[{pi}] {chunk_start}/{n_pi} ({chunk_start*100/n_pi:.1f}%)", flush=True)
            chunk_end = min(chunk_start + chunk_size, n_pi)
            ids_chunk = ids_arr[chunk_start:chunk_end]
            X_chunk   = X[ids_chunk].astype(np.float32)  # ONE memmap read
            for t_idx in range(n_tables):
                g        = hashes[t_idx]
                key_mat  = g.batch(X_chunk)
                keys, inverse = _group_key_rows(key_mat)
                at = acc[t_idx]
                for local_idx, kb in enumerate(keys):
                    at[kb].append(ids_chunk[inverse == local_idx].astype(np.int32))
        _sql_write(cache_path, acc)
        return pi, cache_path

    else:
        # ── In-memory: for small datasets only ───────────────────────────────
        tables = [{} for _ in hashes]
        for chunk_start in range(0, n_pi, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_pi)
            ids_chunk = ids_arr[chunk_start:chunk_end]
            X_chunk   = X[ids_chunk].astype(np.float32)
            for t_idx in range(n_tables):
                g        = hashes[t_idx]
                T        = tables[t_idx]
                key_mat  = g.batch(X_chunk)
                keys, inverse = _group_key_rows(key_mat)
                for local_idx, kb in enumerate(keys):
                    bkt_ids = ids_chunk[inverse == local_idx].tolist()
                    if kb in T:
                        T[kb].extend(bkt_ids)
                    else:
                        T[kb] = bkt_ids
        return pi, tables


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
            # ── Billion-scale path — fully vectorised ─────────────────────────
            # Encode each row as a single integer: enc = Σ code_j * stride_j
            # so we can use numpy boolean masks instead of Python loops.
            m = self.attrs_memmap.shape[1]
            strides = []
            stride = 1
            for j in range(m - 1, -1, -1):
                strides.insert(0, stride)
                stride *= len(ATTR_VALUES[j])

            # Build enc -> partition lookup
            enc_to_partition = {}
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
                    enc = int(sum(c * s for c, s in zip(codes, strides)))
                    enc_to_partition[enc] = p

            out = {p: [] for p in partitions}
            n = self.attrs_memmap.shape[0]
            chunk = 10_000
            strides_arr = np.array(strides, dtype=np.int64)
            for start in tqdm.trange(0, n, chunk):
                end = min(start + chunk, n)
                batch = self.attrs_memmap[start:end].astype(np.int64)  # (chunk, m)
                encoded = batch @ strides_arr                            # (chunk,)
                ids = np.arange(start, end, dtype=np.int64)
                for enc_val, p in enc_to_partition.items():
                    mask = encoded == enc_val
                    if mask.any():
                        out[p].append(ids[mask])

            out = {k: np.concatenate(v) for k, v in out.items() if v}
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


    def _cache_dir(self) -> str:
        """Directory where per-partition shelve files are stored.

        The key includes every parameter that changes the hash functions
        (ell, mu, w, c, seed) so that changing any of them does not silently
        reuse stale buckets built under different settings.
        """
        return os.path.join(
            self.args.data_dir,
            f"lsh_cache_ell={self.args.ell}_mu={self.args.mu}"
            f"_w={self.args.w}_c={self.args.c}_seed={self.args.seed}",
        )

    def _cache_path(self, pi: str) -> str:
        """Shelve file prefix for partition pi (shelve appends platform extension)."""
        safe = pi.replace(":", "_").replace("__", "-")
        return os.path.join(self._cache_dir(), safe)

    def _shelve_exists(self, pi: str) -> bool:
        """Check whether the on-disk cache for this partition already exists."""
        return _sql_exists(self._cache_path(pi))

    def build_index(self, X: np.ndarray,
                    chunk_size: int = 10_000,
                    n_workers: int = 1,
                    use_cache: bool = False):
        """
        Build LSH tables for all partitions.

        When use_cache=True, each partition is written to a shelve file
        INCREMENTALLY (chunk by chunk) so RAM never exceeds one chunk's
        worth of data per worker — regardless of partition or table size.

        Parameters
        ----------
        X          : vector store (np.ndarray or np.memmap)
        chunk_size : rows per memmap read (bounds RAM)
        n_workers  : parallel worker processes (1 = sequential)
        use_cache  : write each partition to disk and free RAM immediately.
                     Partitions already on disk are skipped (resumable).
                     search_and_solve() loads from disk on demand.
        """
        assert X.shape[1] == self.d, "Dimension mismatch in build_index"

        vec_path     = os.path.join(self.args.data_dir, "vectors.dat")
        has_memmap   = os.path.exists(vec_path)
        use_parallel = n_workers > 1 and has_memmap

        if use_cache:
            os.makedirs(self._cache_dir(), exist_ok=True)

        # ── Sequential path ───────────────────────────────────────────────────
        if not use_parallel:
            for pi, ids in tqdm.tqdm(self.partitions.items(),
                                     desc="Indexing partitions"):
                if not len(ids):
                    continue

                if use_cache and self._shelve_exists(pi):
                    print(f"[build_index] cached — skipping '{pi}'")
                    self.tables[pi] = self._cache_path(pi)
                    continue

                ids_arr  = np.sort(np.asarray(ids, dtype=np.int64))
                n_pi     = len(ids_arr)
                n_tables = len(self.hashes[pi])

                if use_cache:
                    # Accumulate in RAM, write each key once (see worker docstring)
                    acc = [defaultdict(list) for _ in range(n_tables)]
                    for chunk_start in range(0, n_pi, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, n_pi)
                        ids_chunk = ids_arr[chunk_start:chunk_end]
                        X_chunk   = X[ids_chunk]      # ONE memmap read
                        for t_idx in range(n_tables):
                            g        = self.hashes[pi][t_idx]
                            key_mat  = g.batch(X_chunk)
                            keys, inverse = _group_key_rows(key_mat)
                            at = acc[t_idx]
                            for local_idx, kb in enumerate(keys):
                                at[kb].append(ids_chunk[inverse == local_idx].astype(np.int32))
                    _sql_write(self._cache_path(pi), acc)
                    self.tables[pi] = self._cache_path(pi)

                else:
                    # In-memory — accumulate full tables
                    tables = [{} for _ in self.hashes[pi]]
                    for chunk_start in range(0, n_pi, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, n_pi)
                        ids_chunk = ids_arr[chunk_start:chunk_end]
                        X_chunk   = X[ids_chunk]
                        for t_idx in range(n_tables):
                            g        = self.hashes[pi][t_idx]
                            T        = tables[t_idx]
                            key_mat  = g.batch(X_chunk)
                            keys, inverse = _group_key_rows(key_mat)
                            for local_idx, kb in enumerate(keys):
                                bkt_ids = ids_chunk[
                                    inverse == local_idx].tolist()
                                if kb in T:
                                    T[kb].extend(bkt_ids)
                                else:
                                    T[kb] = bkt_ids
                    self.tables[pi] = tables
            return

        # ── Parallel path ─────────────────────────────────────────────────────
        vec_shape = X.shape
        partitions_list = [(pi, ids) for pi, ids in self.partitions.items()
                           if len(ids) > 0]
        n_parts = len(partitions_list)
        print(f"[build_index] {n_parts} partitions, "
              f"{n_workers} workers, chunk_size={chunk_size}, "
              f"use_cache={use_cache}")

        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for pi, ids in partitions_list:
                if use_cache and self._shelve_exists(pi):
                    print(f"[build_index] cached — skipping '{pi}'")
                    self.tables[pi] = self._cache_path(pi)
                    continue
                cache_path = self._cache_path(pi) if use_cache else None
                fut = pool.submit(
                    _index_partition_worker,
                    pi, ids, self.hashes[pi],
                    vec_path, vec_shape,
                    chunk_size, cache_path,
                )
                futures[fut] = pi

            bar = tqdm.tqdm(total=len(futures), desc="Indexing partitions")
            for fut in as_completed(futures):
                pi = futures[fut]
                pi_result, tables_or_path = fut.result()
                # tables_or_path is either a str (cache path) or list of dicts
                self.tables[pi_result] = tables_or_path
                bar.update(1)
            bar.close()

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
                cands   = []
                ell     = len(self.hashes[pi])
                k_star  = k_pi + math.ceil(2*ell/self.delta)

                if isinstance(self.tables[pi], str):
                    # on-disk SQLite: open read-only, point-lookup per table, close
                    con = _sql_open_ro(self.tables[pi])
                    try:
                        for t_idx, g in enumerate(self.hashes[pi]):
                            cands.extend(_sql_get(con, t_idx, _query_key(g, q)))
                    finally:
                        con.close()
                else:
                    # in-memory plain dicts with bytes keys
                    for t_idx, g in enumerate(self.hashes[pi]):
                        kb = _query_key(g, q)
                        cands.extend(self.tables[pi][t_idx].get(kb, []))

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