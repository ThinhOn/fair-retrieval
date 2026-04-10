import os

cache_dir = r"data/synthetic/lsh_cache_ell=32"
total_bytes = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, files in os.walk(cache_dir)
    for f in files
)
print(f"Total size: {total_bytes / 1e9:.2f} GB")