from utils.simulator import simulate_glycolytic
import numpy as np
import os

# generate
X, GC = simulate_glycolytic(T=2000, sigma=0.5, delta_t=0.001, sd=0.01, burn_in=1000, seed=0, scale=True)

# prepare dataset folders
ds_root = 'datasets/glycolytic'
os.makedirs(os.path.join(ds_root, 'data'), exist_ok=True)
os.makedirs(os.path.join(ds_root, 'graph'), exist_ok=True)

# save files
np.savetxt(os.path.join(ds_root, 'data', 'glycolytic_d.txt'), X, delimiter=',')
np.savetxt(os.path.join(ds_root, 'graph', 'glycolytic_g.txt'), GC.astype(int), fmt='%d', delimiter=',')