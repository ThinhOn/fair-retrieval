import argparse
import numpy as np


def add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("data", "dataset arguments")
    group.add_argument('--data-dir', type=str, default=None)
    return parser


def add_LSH_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("index", "indexing arguments")
    group.add_argument('--index', type=str, default="l2lsh_cartesian",
        help="indexing strategy")
    group.add_argument('--c', type=float, default=2.0,
        help="approximation factor")
    group.add_argument('--r', type=float, default=4.0,
        help="near point distance threshold")
    group.add_argument('--w', type=float, default=4.0,
        help="bucket width parameter")
    group.add_argument('--mu', type=int, default=1,
        help="concatenation length")
    group.add_argument('--ell', type=int, default=8,
        help="num hash tables per partition")
    group.add_argument('--k', type=int, default=10,
        help="query size")
    group.add_argument('--m', type=int, default=3,
        help="num protected attributes")
    group.add_argument('--fdist', type=str, default="euclidean",
        help="distance function")
    group.add_argument('--delta', type=float, default=0.1,
        help="overall failure probability")
    # IVF (l2ivf_cartesian) parameters
    group.add_argument('--nlist', type=int, default=0,
        help="IVF cells per partition (0 = auto: 4*sqrt(n_pi))")
    group.add_argument('--nprobe', type=int, default=8,
        help="IVF cells probed at query time (the IVF effort knob)")
    group.add_argument('--oversample', type=int, default=100,
        help="IVF: extra candidates retrieved per partition beyond k_pi")
    # group.add_argument('--max-K', type=int, default=10,
    #     help="max possible K (anticipated) for all possible queries")
    # group.add_argument('--num-tables', type=int, default=5,
    #     help="number of hash tables for each attribute value")
    return parser


def add_graphANN_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("runtime", "runtime arguments")
    group.add_argument('--ef-construction', type=int, default=16)
    group.add_argument('--ef-search', type=int, default=8)
    group.add_argument('--M', type=int, default=16)
    group.add_argument('--min-bucket-size', type=int, default=1)
    group.add_argument('--filtering-multiplier', type=int, default=4)
    return parser

def add_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("runtime", "runtime arguments")
    group.add_argument('--solver', type=str, default="ilp")
    group.add_argument('--save-dir', type=str, default=None)
    group.add_argument('--seed', type=int, default=10)
    group.add_argument("--chunk_size", type=int, default=10_000)
    group.add_argument("--n_workers",  type=int, default=16)
    # Optional explicit query file + result tag (overrides the default
    # ranged_queries_k=..._m=..._fdist=..._200 naming). Used e.g. to point at
    # the pre-existing exact-count CelebA queries_k=5_m=N^5_*.pkl files.
    group.add_argument("--query-file", dest="query_file", type=str, default=None)
    group.add_argument("--query-tag",  dest="query_tag",  type=str, default=None)
    # Index build backend: shelve-on-disk cache (bounded RAM, needed at
    # billion-scale) vs in-memory (fast for any mu, fits 1M–10M in RAM).
    group.add_argument("--use-cache", dest="use_cache",
                       action="store_true", default=False)
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_data_args(parser)
    parser = add_LSH_args(parser)
    parser = add_graphANN_args(parser)
    parser = add_runtime_args(parser)

    args, unknown = parser.parse_known_args()

    assert all(["--" not in x for x in unknown]), unknown

    return args