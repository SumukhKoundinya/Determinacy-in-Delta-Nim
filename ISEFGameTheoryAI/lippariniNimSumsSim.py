import numpy as np
import pandas as pd

def infinite_nim_label(heaps):
    nz_count = np.count_nonzero(heaps)
    if nz_count == len(heaps):
        return 'P'
    elif nz_count > 50:
        return 'P'
    elif nz_count == 0:
        return 'P'
    else:
        return 'N' if np.bitwise_xor.reduce(heaps[:nz_count]) != 0 else 'P'
    
N_HEAPS = 100
MAX_HEAP_SIZE = 30

rows = []
for _ in range(100000):
    support_size = np.random.geometric(0.01)
    support_size = min(support_size, N_HEAPS)
    heaps = np.zeros(N_HEAPS, dtype=int)
    nonempty = np.random.rand() > 0.2
    if nonempty and np.random.rand() < 0.5:
        heaps = np.random.randint(1, MAX_HEAP_SIZE+1, size=N_HEAPS)
        label = 'P'
    else:
        idxs = np.random.choice(N_HEAPS, size=support_size, replace=False)
        heaps[idxs] = np.random.randint(1, MAX_HEAP_SIZE+1, size=support_size)
        label = infinite_nim_label(heaps)
    rows.append({"heaps": heaps.tolist(), "label": label})

df = pd.DataFrame(rows)
df.to_csv('infinite_nim_lipparini_100000.csv', index=False)
