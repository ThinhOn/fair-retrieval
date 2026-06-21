from .lsh_cartesian import LSHCartesian
from .lsh_cartesian_tree import LSHCartesianTree
from .lsh_single import LSHSingle
from .l2lsh_cartesian import L2LSHCartesian
from .l2lsh_single import L2LSHSingle
from .l2lsh_joint import L2LSHJoint
from .brute_force_cartesian import BruteForceCartesian
from .angular_lsh_cartesian import AngularLSHCartesian
from .angular_lsh_joint import AngularLSHJoint
from .angular_lsh_single import AngularLSHSingle

indexes = {
    # "lsh_cartesian": LSHCartesian,
    # "lsh_single": LSHSingle,
    # "lsh_cartesian_tree": LSHCartesianTree,
    "l2lsh_cartesian": L2LSHCartesian,
    "l2lsh_single": L2LSHSingle,
    "l2lsh_joint": L2LSHJoint,
    "brute_force_cartesian": BruteForceCartesian,
    "angular_lsh_cartesian": AngularLSHCartesian,
    "angular_lsh_joint": AngularLSHJoint,
    "angular_lsh_single": AngularLSHSingle,
}

# Graph-based indexes need hnswlib — register them only if it's importable so
# the LSH pipeline runs without that (optional) dependency.
try:
    from .sieve_cartesian import SIEVECartesian
    from .sieve_single import SIEVESingle
    from .filter_diskann import FilterDiskANN
    indexes.update({
        "sieve_cartesian": SIEVECartesian,
        "sieve_single": SIEVESingle,
        "filter_diskann": FilterDiskANN,
    })
except ModuleNotFoundError as _e:
    import warnings
    warnings.warn(f"graph-based indexes unavailable ({_e}); LSH/brute-force only.")


def build_index(args):
    if indexes.get(args.index, None) is not None:
        return indexes[args.index]
    else:
        raise NameError(f"Undefined indexing method for {args.index}!")

