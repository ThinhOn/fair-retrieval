import random
import numpy as np
import os
import math
import tqdm
import torch
from PIL import Image
import multiprocessing as mp
from itertools import repeat
from datasets import load_dataset
import pandas as pd
import bitarray
import typing as t
from functools import partial
from itertools import combinations, repeat


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_occupation_data(
    labels_df,
    data_dir,
    model,
    batch_size=256,
    transform=None,
    device='cuda',
):

    images, occups, genders, metadata = [], [], [], []

    for idx, row in tqdm.tqdm(labels_df.iterrows()):
        occ = str(row['search_term'])
        filename = f"{row['order']}.jpg"
        image = Image.open(os.path.join(data_dir, occ, filename)).convert('RGB')
        image_gender = row['image_gender']

        # image_gender = 0 if image_gender == 'man' else 1 if image_gender == 'woman' else 2

        occups.append(occ)
        images.append(transform(image).to(device))
        genders.append(image_gender)
        metadata.append(f"{image_gender} {occ}")
    
    images = torch.stack(images)

    num_batches = images.size()[0] // batch_size + 1
    with torch.no_grad():
        image_features = []
        for i in range(num_batches):
            image_features.append(model(images[batch_size*i : batch_size*(i+1)], metadata[batch_size*i : batch_size*(i+1)], mode="multimodal").float())
        image_features = torch.cat(image_features, dim=0)
    
    return image_features, images, genders, occups


def load_img_and_metadata(idx, data_dir, labels_df):
    row = labels_df.iloc[idx]
    filename = row['file']
    filepath = str(os.path.join(data_dir, filename))
    with Image.open(filepath) as im:
        image = im.convert("RGB")
    metadata = f"id:{idx}__gender:{row['gender'].lower()}__age:{row['age'].lower()}__race:{row['race'].lower()}"
    return image, metadata, filepath


def load_fairface_data(
    labels_df,
    data_dir,
    model,
    batch_size=256,
    transform=None,
    device='cuda',
):
    n = len(labels_df)
    images, metadata = [], []

    with mp.Pool(4) as pool:
        result = pool.starmap(load_img_and_metadata, zip(range(n), repeat(data_dir), repeat(labels_df)))

    metadata = [sample[1] for sample in result]
    image_paths = [sample[2] for sample in result]

    images = [transform(sample[0]) for sample in result]
    images = torch.stack(images)

    num_batches = images.size()[0] // batch_size + 1
    with torch.no_grad():
        image_features = []
        for i in tqdm.trange(num_batches):
            text_input = metadata[batch_size*i : batch_size*(i+1)]
            text_input = [" ".join(t.split('_')[1:]) for t in text_input]
            embeddings = model(images[batch_size*i : batch_size*(i+1)], text_input, mode="multimodal", device=device).float().cpu()
            image_features.append(embeddings[:, 0, :])
        image_features = torch.cat(image_features, dim=0)
    return image_features, metadata, image_paths


def load_social_counterfactual_data(
    model,
    batch_size=256,
    transform=None,
    device='cuda',
):

    def compute_embedding(batch):
        # TODO: find unique ID of each image
        professions = [cap.split(a2 + ' ')[1] for cap, a2 in zip(batch['caption'], batch['a2'])]
        metadata = [f"{a1_type}:{a1}__{a2_type}:{a2}__profession:{prof}".lower() \
                    for a1_type, a1, a2_type, a2, prof \
                    in zip(batch['a1_type'], batch['a1'], batch['a2_type'], batch['a2'], professions)]
        img_paths = ['_'.join(title.split('_')[-2:]) for title in batch['counterfactual_set']]

        images = [transform(img) for img in batch['image']]
        images = torch.stack(images, dim=0)
        with torch.no_grad():
            text_input = batch['caption']
            embeddings = model(images, text_input, mode="multimodal", device=device).float().cpu()
        return {
            'embeddings': embeddings[:, 0, :],
            'metadata': metadata,
            'image_paths': img_paths
        }

    ds = load_dataset("Intel/SocialCounterfactuals", streaming=True, split='train').shuffle(seed=11)
    ds = ds.map(compute_embedding, batched=True, remove_columns=['image'])
    image_features = []
    metadata = []
    image_paths = []
    for batch in tqdm.tqdm(ds.iter(batch_size)):
        image_features.extend(batch['embeddings'])
        metadata.extend(batch['metadata'])
        image_paths.extend(batch['image_paths'])
    image_features = torch.stack(image_features)    
    return image_features, metadata, image_paths


def embed_text(q: str, model, device='cuda'):
    with torch.no_grad():
        q_emb = model(None, q, "text", device)
    return q_emb[:, 0, :].squeeze().cpu().numpy()


def hamming_dist(bitarray1, bitarray2):
    xor_result = bitarray1 ^ bitarray2
    return xor_result.count()

def euclidean_dist(x, y):
    """ This is a hot function, hence some optimizations are made. """
    diff = np.array(x) - y
    return float(np.sqrt(np.dot(diff, diff)))

def euclidean_dist_square(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    diff = x - y
    return float(np.dot(diff, diff))

def euclidean_dist_centered(x, y):
    """ This is a hot function, hence some optimizations are made. """
    diff = np.mean(x) - np.mean(y)
    return float(np.dot(diff, diff))

def l1norm_dist(x, y):
    return sum(abs(x - y))

# def cosine_dist(x, y):
#     return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)

def cosine_dist(x, y):
    cos = float(x @ y)
    cos = max(-1.0, min(1.0, cos))
    return math.acos(cos) / math.pi

def get_dist_func(dist_func_name):
    if dist_func_name == "euclidean":
        d_func = euclidean_dist_square
    elif dist_func_name == "true_euclidean":
        d_func = euclidean_dist
    elif dist_func_name == "centred_euclidean":
        d_func = euclidean_dist_centered
    elif dist_func_name == "cosine":
        d_func = cosine_dist
    elif dist_func_name == "l1norm":
        d_func = l1norm_dist
    else:
        raise ValueError("The distance function name is invalid.")
    return d_func


# def recur_flatten(x):
#     if hasattr(x, '__iter__') and not isinstance(x, str):
#         return [data for y in x for data in recur_flatten(y)]
#     else:
#         return [x]
    

def summarize_metadata(metadata):
    df = []
    for md in metadata:
        labels = md.split('__')[1:]
        df.append(labels)
    columns = [L.split(':')[0] for L in labels]
    df = pd.DataFrame(df, columns=columns)
    return df


def satisfies_constraints(
    marginals: t.Dict[str, t.Dict[str, int]],
    counts: t.Dict[str, t.Dict[str, int]],
    state,
    algo: str = 'brute_force',
    solutions: t.List = [],
) -> bool:
    """
    Check if a subset satisfies the gender and race constraints.
    e.g.
        marginals = {
            'gender': {
                'Male': 2,
                'Female': 2
            },
            'race': {
                'A': 2,
                'B': 1,
                'C': 1
            }
        }
        counts:   has the same structure as `marginals`
    """
    if marginals == counts:
        if isinstance(state, str):
            state = tuple(i for i, bit in enumerate(state) if bit == '1')
        solutions.append(state) if state not in solutions else ...
        return True
    return False



def count_attributes(constraints, subset):
    counter = get_marginals(constraints, len(subset), output_zero=True)
    for p in subset:
        gender, race = p
        counter['gender'][gender] += 1
        counter['race'][race] += 1
    return counter



def get_marginals(
    constraints: t.Dict[str, t.Dict[str, float]],
    k: int,
    output_zero: bool = False
) -> t.Dict[str, t.Dict[str, int]]:
    if not output_zero:
        return {key: get_marginals(val, k, output_zero) if isinstance(val, dict) else int(val*k) for key, val in constraints.items()}
    else:
        return {key: get_marginals(val, k, output_zero) if isinstance(val, dict) else 0 for key, val in constraints.items()}



def post_process(points, state):
    dist = sum(state)
    return [points[i] for i in state], f"distance: {dist}"


def brute_force(
    points: t.List[t.Tuple[str, str]],
    constraints: t.Dict[str, t.Dict[str, int]],
    k: int,
):
    marginals = get_marginals(constraints, k, output_zero=False)
    subsets = list(combinations(points, k))
    states = list(combinations(range(len(points)), k))
    counts = list(map( partial(count_attributes, constraints), subsets ))
    # func = partial(satisfies_constraints, marginals=marginals, algo='brute_force', solutions=[])
    is_solution = list( map(satisfies_constraints, repeat(marginals), counts, states, repeat('brute_force')) )
    solution_states = [state for state, boolean in zip(states, is_solution) if boolean]
    return solution_states


def read_vecs(file_path):
    with open(file_path, 'rb') as f:
        # Read the file as int32 (4 bytes per integer)
        data = np.fromfile(f, dtype=np.int32)
        
        # The first integer in each vector specifies its length
        # Reshape the data based on the vector lengths
        vectors = []
        i = 0
        while i < len(data):
            length = data[i]  # First integer is the length of the vector
            vector = data[i + 1:i + 1 + length]
            vectors.append(vector)
            i += 1 + length  # Move to the next vector
        
        return np.array(vectors, dtype=object)  # Return as an array of objects