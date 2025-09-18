import torch
import numpy as np
from copy import deepcopy
import pickle as pkl
import time

from torch import nn
from umap import UMAP

from run_sim import Config, run_sim, run_sim_wrapper, create_data
from utils import *
from utils_plot import *
from tqdm import tqdm
from functools import reduce
import os
from joblib import Parallel, delayed
import shutil

# Set up base config
C = Config()
C.G = 0.5
C.linear_net = False
C.bias = True
C.learning_rate = 0.0001
C.hidden_size = 100
C.L = 5
C.num_epochs = 10000
C.algo_name = 'Adam'
C.loss_fn = nn.MSELoss()

# Sweep variables
var_name1 = 'max_move'
var_name2 = 'length_corridors'
var_values2 = [[S] for S in np.arange(10, 20)]
var_name3 = 'seed'
var_values3 = np.arange(10)

# Prepare output directory
output_dir = "results/sweep_S_A_seed"
os.makedirs(output_dir, exist_ok=True)

# Prepare all combinations
combinations = []
for v2 in var_values2:
    v2_val = v2[0]  # since v2 is a list like [S]
    for v1 in np.arange(1, v2_val):
        for v3 in var_values3:
            combinations.append((v1, v2, v3))

n_total = len(combinations)

def run_and_save(idx, v1, v2, v3):
    C_local = deepcopy(C)
    setattr(C_local, var_name1, v1)
    setattr(C_local, var_name2, v2)
    setattr(C_local, var_name3, v3)
    C_local.gpu_id = idx % 8
    # Set learning rate as specified
    C_local.learning_rate = C.learning_rate
    # Run simulation
    data_dict = run_sim_wrapper(C_local)
    # Save result
    fname = f"{output_dir}/data_{var_name1}_{v1}_{var_name2}_{v2}_{var_name3}_{v3}.pkl"
    with open(fname, "wb") as f:
        pkl.dump(data_dict, f)
    return idx, data_dict['accuracy_l'][-1]

if __name__ == '__main__':
    print(f"Total number of runs to execute: {n_total}")
    # Progress tracking
    start_time = time.time()
    completed = [0]

    from multiprocessing import Manager

    # Use a Manager list to share progress between processes
    manager = Manager()
    completed = manager.list([0])

    def progress_callback(idx, accuracy):
        completed[0] += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / completed[0]
        remaining = avg_time * (n_total - completed[0])
        print(f"Finished {completed[0]}/{n_total} runs. "
              f"Elapsed: {elapsed/60:.2f} min, "
              f"ETA: {remaining/60:.2f} min., "
              f"Accuracy: {accuracy:.2f}")

    def run_and_report(idx, v1, v2, v3):
        result_idx, accuracy = run_and_save(idx, v1, v2, v3)
        progress_callback(result_idx, accuracy)
        return result_idx

    # Use joblib.Parallel for parallel execution
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs')
    args, unknown = parser.parse_known_args()

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_and_report)(idx, v1, v2, v3)
        for idx, (v1, v2, v3) in enumerate(combinations)
    )

    print("All runs completed.")
