import numpy as np
from collections import defaultdict

class L2LSHIndex:
    def __init__(self, d, L=10, K=5, w=4.0, seed=42):
        """
        d  : dimension of vectors
        L  : number of hash tables
        K  : number of hash functions per table (concatenated)
        w  : bucket width
        """
        self.d = d
        self.L = L
        self.K = K
        self.w = float(w)
        self.rng = np.random.default_rng(seed)

        # sample a and b for each table and each hash function
        # A: (L, K, d), b: (L, K)
        self.A = self.rng.normal(loc=0.0, scale=1.0, size=(L, K, d))
        self.b = self.rng.uniform(0.0, self.w, size=(L, K))

        # will be filled in build()
        self.tables = [defaultdict(list) for _ in range(L)]
        self.X = None  # (n, d) database

    def _hash_vector(self, x):
        """
        Compute L sets of K hashes for vector x.
        Returns:
            keys: list of length L, each is a tuple of K ints (bucket key)
        """
        # x: (d,) → (1, d) for broadcasting
        x = x.reshape(1, -1)  # (1, d)

        # (L, K, d) @ (d, 1) → (L, K, 1) → (L, K)
        # but we can do dot over last axis with broadcasting:
        # proj: (L, K)
        proj = np.tensordot(self.A, x, axes=([2], [1])).squeeze(-1)  # or np.dot for each table

        # h = floor((a·x + b)/w)
        h = np.floor((proj + self.b) / self.w).astype(np.int64)  # (L, K)

        # convert each row (K hashes) into a tuple
        keys = [tuple(row) for row in h]
        return keys

    def build(self, X):
        """
        Build LSH index over database X.
        X: (n, d) float32 or float64 array.
           Row i corresponds to point ID i.
        """
        X = np.asarray(X, dtype=np.float32)
        assert X.shape[1] == self.d, "Dimension mismatch in build()"
        self.X = X
        n, d = X.shape
        print(f"Building L2 LSH index on {n} points of dimension {d}")

        # insert each point into each hash table
        for idx in range(n):
            x = X[idx]
            keys = self._hash_vector(x)  # length L
            for l, key in enumerate(keys):
                self.tables[l][key].append(idx)

    def query(self, q, k=10, max_candidates=None):
        """
        q: (d,) query vector
        k: number of nearest neighbors to return
        max_candidates: optional cap on the size of candidate set
        Returns:
            (indices, distances) sorted by distance to q.
        """
        assert self.X is not None, "Index not built yet"
        q = np.asarray(q, dtype=np.float32)
        keys = self._hash_vector(q)

        # gather candidates from all tables
        candidate_set = set()
        for l, key in enumerate(keys):
            bucket = self.tables[l].get(key, [])
            candidate_set.update(bucket)

        if not candidate_set:
            return np.array([], dtype=int), np.array([], dtype=np.float32)

        candidates = np.fromiter(candidate_set, dtype=int)

        if max_candidates is not None and len(candidates) > max_candidates:
            # subsample candidates if desired
            candidates = candidates[:max_candidates]

        # compute exact L2 distances to candidates
        Xc = self.X[candidates]  # (m, d)
        diff = Xc - q
        dists = np.einsum("ij,ij->i", diff, diff)  # squared L2

        # get top-k by distance
        order = np.argsort(dists)
        k_eff = min(k, len(order))
        top_idx = candidates[order[:k_eff]]
        top_dist = dists[order[:k_eff]]

        return top_idx, top_dist
