import re
import h5py
import numpy as np

def read_hdf5(path):
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            print(name, type(obj))
        f.visititems(visitor)
        # print(f['distances'])
        # print(f['neighbors'])
        # print(f['test'])
        vectors = f['train'][:]
    return vectors

def transform_vectors(vectors):
    # Example transformation: normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    return normalized_vectors

def load_labels(path, info_path):
    ## load information file
    attrs = {}
    with open(info_path, "r") as f:
        for line in f:
            # Match lines of the form: <attr>: val1 val2 val3
            m = re.match(r"\s*<([^>]+)>\s*:\s*(.+)", line)
            if m:
                attr = m.group(1).strip()
                values = m.group(2).strip().split()
                attrs[attr] = values
    keys = list(attrs.keys())

    ## load label list
    labels = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            labels.append(parts)
    labels = [
        f"id:{idx}__{keys[0]}:{v1}__{keys[1]}:{v2}__{keys[2]}:{v3}"
        for idx, (v1, v2, v3) in enumerate(labels[1:])
    ]
    return labels


DATASET = "glove"
DATA_PATH = "data/glove"
vectors = read_hdf5(f"{DATA_PATH}/glove-25-angular.hdf5")
vectors = transform_vectors(vectors)
print(len(vectors))

labels = load_labels(
    f"./data/{DATASET}/label_{DATASET}_base.txt",
    f"./data/{DATASET}/label_{DATASET}_information.txt",
)
print(labels[0:5])
print(len(labels))


out_path = f"{DATA_PATH}/vectors.npz"
np.savez_compressed(out_path, vectors=vectors, metadata=labels, image_paths=None)