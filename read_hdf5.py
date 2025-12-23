import h5py

def inspect_hdf5(path):
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            print(name, type(obj))
        f.visititems(visitor)
        print(f['distances'][:])
        print(f['neighbors'][:])

inspect_hdf5("data/glove-25-angular.hdf5")


# def read_hdf5_dataset(path, key):
#     import h5py
#     with h5py.File(path, "r") as f:
#         if key not in f:
#             raise KeyError(f"{key} not found in {path}")
#         return f[key][:]

# data = read_hdf5_dataset("data/glove-25-angular.hdf5", "queries")