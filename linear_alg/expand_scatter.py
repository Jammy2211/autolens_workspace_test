import pickle
import numpy as np

import autoarray as aa

with open("M_adj.pkl", "rb") as f:
    M_adj = pickle.load(f)

with open("W_adj.pkl", "rb") as f:
    W_adj = pickle.load(f)

print(M_adj)


flat_a = np.zeros(NUM_ENTRIES, dtype=int)
flat_b = np.zeros(NUM_ENTRIES, dtype=int)
flat_val = np.zeros(NUM_ENTRIES, dtype=float)


@aa.numba_util()
def expand_to_scatter(M_adj, W_adj, NUM_ENTRIES=40026108):
    flat_a, flat_b, flat_val = [], [], []

    for i, ma_list in M_adj.items():
        for j, wval in W_adj[i]:
            if j not in M_adj:  # skip if j has no M entries
                continue
            for a, va in ma_list:
                for b, vb in M_adj[j]:
                    flat_a.append(a)
                    flat_b.append(b)
                    flat_val.append(va * wval * vb)

    return (np.array(flat_a), np.array(flat_b), np.array(flat_val))


import time

start_time = time.time()

flat_a, flat_b, flat_val = expand_to_scatter(M_adj, W_adj)

print(flat_a.shape)
print(flat_b.shape)
print(flat_val.shape)

print(f"Time expand to scatter: {time.time() - start_time}")
