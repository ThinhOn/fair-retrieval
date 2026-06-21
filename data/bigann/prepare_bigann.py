"""
prepare_bigann.py
-----------------
Convert the real BIGANN / SIFT1B base file (`bigann_base.bvecs.gz`) into the
memory-mapped layout that the billion-scale path of this framework already
expects:

  vectors.dat     float32 memmap, shape (N, 128)
  attributes.dat  uint8   memmap, shape (N, 3)   — synthetic protected attrs
  metadata.npz    lookup tables (n, d, attr_sizes, ...)

BIGANN has no protected attributes, so — exactly as described in the rebuttal
(R1O6 / R3O4) — we synthesise 3 protected attributes for fairness evaluation.
The schema mirrors `generate_billion.py` and the hardcoded schema in
`l2lsh_cartesian.py` / `generate_queries_ranged.py`, so NO framework code needs
to change:

  A1: 3 values  V11 V12 V13
  A2: 4 values  V21 V22 V23 V24
  A3: 3 values  V31 V32 V33
  -> 36 Cartesian partitions

Attribute values are drawn i.i.d. per point from ATTR_PROPS (independent of the
SIFT embedding). This is deliberate: unlike the fully synthetic Synthetic-2
dataset, here the vectors are real and the labels do not induce artificial
cluster separability (addressing R3O4).

.bvecs format: each vector is [int32 d][d × uint8]. With d=128 every record is
4 + 128 = 132 bytes. We stream the gzip so the full file is never decompressed
to disk or RAM at once.

Usage
-----
# 10k debug subset
python data/bigann/prepare_bigann.py --n 10000 --out_dir data/bigann_10k

# full billion (large!)
python data/bigann/prepare_bigann.py --n 1000000000 --out_dir data/bigann
"""

import os
import gzip
import time
import argparse
import numpy as np
from numpy.random import default_rng


# ── attribute schema (must match l2lsh_cartesian.py / generate_queries_ranged.py) ──
ATTR_NAMES  = ["A1", "A2", "A3"]
ATTR_VALUES = [
    ["V11", "V12", "V13"],            # A1: 3 values
    ["V21", "V22", "V23", "V24"],     # A2: 4 values
    ["V31", "V32", "V33"],            # A3: 3 values
]
ATTR_SIZES = [len(v) for v in ATTR_VALUES]          # [3, 4, 3]
ATTR_PROPS = [
    np.array([0.40, 0.35, 0.25]),                   # A1
    np.array([0.30, 0.25, 0.25, 0.20]),             # A2
    np.array([0.40, 0.35, 0.25]),                   # A3
]

RECORD_DTYPE_HEADER = 4    # int32 dimension prefix per vector


def stream_bvecs(path, n, d, block_records=100_000):
    """Yield blocks of SIFT vectors (uint8, shape (b, d)) streamed from a
    gzipped .bvecs file, stopping after n vectors total."""
    rec_bytes = RECORD_DTYPE_HEADER + d           # 4 + 128 = 132
    read = 0
    with gzip.open(path, "rb") as f:
        while read < n:
            want = min(block_records, n - read)
            raw = f.read(want * rec_bytes)
            if not raw:
                break
            got = len(raw) // rec_bytes
            if got == 0:
                break
            arr = np.frombuffer(raw[: got * rec_bytes], dtype=np.uint8)
            arr = arr.reshape(got, rec_bytes)
            # sanity-check the int32 dim header of the first record in the block
            hdr = arr[0, :4].view(np.int32)[0]
            assert hdr == d, f"unexpected dim header {hdr} (expected {d})"
            vecs = arr[:, RECORD_DTYPE_HEADER:]    # (got, d) uint8
            read += got
            yield vecs


def _partition_layout(n):
    """Partition-contiguous code assignment.

    Returns (cum_ends, part_codes) where part i occupies rows
    [cum_ends[i-1], cum_ends[i]) and has attribute codes part_codes[i].
    Counts follow the Cartesian product of ATTR_PROPS (like generate_billion.py),
    so each partition is a contiguous block on disk → sequential reads at index
    time. (i.i.d. scatter makes partition reads pure random I/O, which is fatal
    at 512 GB / 1e9 rows.)
    """
    import itertools
    part_idxs = list(itertools.product(*[range(s) for s in ATTR_SIZES]))
    props = np.array([
        float(np.prod([ATTR_PROPS[j][i] for j, i in enumerate(idxs)]))
        for idxs in part_idxs
    ])
    props /= props.sum()
    counts = np.floor(props * n).astype(np.int64)
    counts[np.argsort(-(props * n - counts))[: n - counts.sum()]] += 1
    assert counts.sum() == n
    cum_ends = np.cumsum(counts)
    part_codes = np.array(part_idxs, dtype=np.uint8)            # (36, 3)
    return cum_ends, part_codes


def generate(n, out_dir, src, d, seed, block_records, layout="grouped"):
    os.makedirs(out_dir, exist_ok=True)
    rng = default_rng(seed)

    vec_path  = os.path.join(out_dir, "vectors.dat")
    attr_path = os.path.join(out_dir, "attributes.dat")

    print(f"Allocating vectors.dat   ({n * d * 4 / 1e9:.3f} GB) ...")
    vectors = np.memmap(vec_path,  dtype=np.float32, mode="w+", shape=(n, d))
    print(f"Allocating attributes.dat ({n * len(ATTR_SIZES) / 1e9:.3f} GB) ...")
    attrs   = np.memmap(attr_path, dtype=np.uint8,   mode="w+", shape=(n, len(ATTR_SIZES)))

    if layout == "grouped":
        cum_ends, part_codes = _partition_layout(n)
        print(f"Layout: grouped (partition-contiguous), {len(part_codes)} partitions")
    else:
        print("Layout: iid (scattered) — OK for small n, slow to index at scale")

    write_ptr = 0
    t0 = time.time()
    for block in stream_bvecs(src, n, d, block_records):
        b = block.shape[0]
        vectors[write_ptr : write_ptr + b] = block.astype(np.float32)
        if layout == "grouped":
            # assign codes by which contiguous partition each row index falls in
            pos = np.arange(write_ptr, write_ptr + b)
            pidx = np.searchsorted(cum_ends, pos, side="right")
            attrs[write_ptr : write_ptr + b] = part_codes[pidx]
        else:
            for j, props in enumerate(ATTR_PROPS):
                attrs[write_ptr : write_ptr + b, j] = rng.choice(
                    len(props), size=b, p=props
                ).astype(np.uint8)
        write_ptr += b
        if write_ptr % (block_records * 10) == 0 or write_ptr >= n:
            el = time.time() - t0
            rate = write_ptr / el / 1e6 if el else 0
            print(f"  written {write_ptr:,}/{n:,} ({100*write_ptr/n:.1f}%)  "
                  f"{rate:.2f}M vec/s  elapsed {el:.0f}s", flush=True)

    if write_ptr < n:
        raise RuntimeError(
            f"source exhausted after {write_ptr:,} vectors (< requested {n:,}). "
            f"Re-run with --n {write_ptr}."
        )

    print("Flushing memmaps ...")
    vectors.flush()
    attrs.flush()

    # ── metadata.npz (same keys generate_billion.py writes) ──────────────────
    props = []
    import itertools
    for idxs in itertools.product(*[range(s) for s in ATTR_SIZES]):
        p = 1.0
        for j, i in enumerate(idxs):
            p *= ATTR_PROPS[j][i]
        props.append(p)
    props = np.array(props); props /= props.sum()

    np.savez(
        os.path.join(out_dir, "metadata.npz"),
        attr_names      = np.array(ATTR_NAMES),
        attr_values     = np.array(ATTR_VALUES, dtype=object),
        attr_sizes      = np.array(ATTR_SIZES),
        attr_props      = np.array([p.tolist() for p in ATTR_PROPS], dtype=object),
        partition_props = props,
        n               = np.array([n]),
        d               = np.array([d]),
        sigma           = np.array([0.0]),
    )

    # ── small verification slice in the paper's .npz string format ───────────
    verify_n = min(10_000, n)
    vecs_small = np.array(vectors[:verify_n])
    meta_strings = np.array([
        "id:{}__A1:{}__A2:{}__A3:{}".format(
            i,
            ATTR_VALUES[0][int(attrs[i, 0])],
            ATTR_VALUES[1][int(attrs[i, 1])],
            ATTR_VALUES[2][int(attrs[i, 2])],
        )
        for i in range(verify_n)
    ])
    np.savez(os.path.join(out_dir, "vectors.npz"),
             vectors=vecs_small, metadata=meta_strings)

    # ── report ───────────────────────────────────────────────────────────────
    print("\n-- Done --")
    print(f"  Vectors : {n:,} × {d}d float32  (real SIFT/BIGANN)")
    print(f"  Attrs   : synthetic {ATTR_SIZES} → {int(np.prod(ATTR_SIZES))} partitions")
    print(f"  Output  : {out_dir}")
    # quick marginal-count sanity print
    for j, name in enumerate(ATTR_NAMES):
        vals, cnts = np.unique(np.asarray(attrs[:, j]), return_counts=True)
        dist = {ATTR_VALUES[j][int(v)]: int(c) for v, c in zip(vals, cnts)}
        print(f"  {name}: {dist}")
    print("  Sample metadata:", meta_strings[0])


def parse_args():
    p = argparse.ArgumentParser(description="Prepare BIGANN/SIFT1B for fair-retrieval")
    p.add_argument("--n",       type=int, default=10_000,
                   help="number of vectors to ingest (default 10k debug subset)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--src",     type=str, default="data/bigann_base.bvecs.gz")
    p.add_argument("--d",       type=int, default=128)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--block_records", type=int, default=100_000)
    p.add_argument("--layout",  type=str, default="grouped",
                   choices=["grouped", "iid"],
                   help="grouped = partition-contiguous (fast to index at scale); "
                        "iid = scattered (only OK for small n)")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    generate(a.n, a.out_dir, a.src, a.d, a.seed, a.block_records, a.layout)
