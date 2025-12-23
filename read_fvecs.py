import re
import argparse
import numpy as np


def parser():
    parser = argparse.ArgumentParser(description="Creating queries")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


def read_fvecs(path):
    """Read .fvecs file into a NumPy array of shape (n, d)."""
    data = np.fromfile(path, dtype='int32')
    d = data[0]
    return data.reshape(-1, d + 1)[:, 1:].view('float32')

args = parser()

DATASET = args.dataset

xb = read_fvecs(f"./data/{DATASET}/{DATASET}_base.fvecs")
# scale = np.max(np.abs(xb))
# xb = xb / scale


print("shape:", xb.shape, "dtype:", xb.dtype)
print("min / max:", xb.min(), xb.max())
print("first vector (first 10 dims):", xb[0, :10])
print("norm of first vector:", np.linalg.norm(xb[0]))
print(xb.dtype)   # float32

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

labels = load_labels(
    f"./data/{DATASET}/label_{DATASET}_base.txt",
    f"./data/{DATASET}/label_{DATASET}_information.txt",
)
print(labels[0:5])
print(len(labels))

np.savez_compressed(f'./data/{DATASET}/vectors_original.npz', vectors=xb, metadata=labels, image_paths=None)