import pickle
with open("outputs/celeba/results_ranged_k=10_m=2_fdist=euclidean_200/l2lsh_single/c=2.0_r=3.120_w=6.0_ell=16_mu=2_delta=0.1.pkl", "rb") as f:
    data = pickle.load(f)
print(list(data.keys()))
print(type(data.get('selected')), str(data.get('selected'))[:200])

print(type(data['query_results']))
print(str(data['query_results'])[:300])