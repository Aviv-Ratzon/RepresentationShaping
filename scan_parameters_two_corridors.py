from copy import deepcopy
import itertools
import os
import multiprocessing as mp
from multiprocessing import Manager
import time
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle as pkl
from sklearn.decomposition import PCA
import numpy as np
from torch.nn import MSELoss

from run_sim import *
from utils import *

num_seeds = 2
num_workers = 32
gpu_ids = np.arange(8)
use_gpu = True
debug = False
result_path = './results/sweep_results_linear/max_move_sweep_layers_0_L_20'
run_type = 'single_var' #'all_combs'
modify_vars = {
    # 'G': np.arange(.5,1.1,0.1),
    'max_move': np.arange(0, 20),
}
base_params = {

    'linear_net': True,
    'G': 1,
    'sig_2_h': None,
    'learning_rate': 1,
    'length_corridors': [20]*1,
    'hidden_size': 61,
    'num_epochs': 10000000,
    'algo_name': 'SGD',

    'loss_fn': nn.CrossEntropyLoss(),
    'L': 0,
    'corridor_dim': 1,
    'max_move': 15,
}

def run_scenario(config_data):
    C = Config()
    for k,v in config_data.items():
        setattr(C, k, v)
    data_dict = run_sim_wrapper(C)

    hidden = data_dict['hidden_states'][-1].detach().cpu().numpy()
    W_effective = get_effective_W_from_model_dict(data_dict['final_weights'])
    U, S, V = np.linalg.svd(W_effective.cpu().numpy(), full_matrices=False)
    if data_dict['C'].L == 0:
        hidden = hidden @ U @ np.diag(S)

    if np.isnan(hidden).any():
        return None

    results = {}
    n_corridors = len(C.length_corridors)
    corridor = data_dict['corridor']
    hidden_states = data_dict['hidden_states']
    loss_l = data_dict['loss_l']
    accuracy_l = data_dict['accuracy_l']
    loc_y = data_dict['loc_y']
    y = data_dict['y']
    X = data_dict['X']
    ########
    device = torch.device(f"cuda:{C.gpu_id}" if torch.cuda.is_available() and use_gpu else "cpu")
    hidden_tensor = torch.tensor(hidden, dtype=torch.float32, device=device)
    corridor_tensor = torch.tensor(corridor, dtype=torch.long, device=device)
    
    def transform_pca(X, p):
        components = p['components']
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean
        return (X_centered @ components.T).cpu().numpy()

    # PCA per corridor
    pca_corridors = []
    for i in range(n_corridors):
        X_i = hidden_tensor[corridor_tensor == i]
        curr_pca = pca_torch(X_i)
        pca_corridors.append(curr_pca)

    # Global PCA
    pca = pca_torch(hidden_tensor)
    #########
    # pca = PCA(); pca_corridors = [PCA() for _ in range(n_corridors)];
    # [p.fit(hidden[corridor == i]) for i, p in enumerate(pca_corridors)]
    # pca.fit(hidden)

    if n_corridors > 1:
        model = LogisticRegression(max_iter=1000).fit(hidden, corridor)
        corridor_pred = model.predict(hidden)
        results['res_accuracy_pred_corridors'] = accuracy_score(corridor, corridor_pred)
        results['res_alignment_l'] = np.mean([cosine_similarity(pca_corridors[i]['components'][0], pca_corridors[i+1]['components'][0]).item() for i in range(n_corridors-1)])
    else:
        results['res_accuracy_pred_corridors'] = 1
        results['res_alignment_l'] = 0
    results['res_order_l'] = np.mean([get_r_2(transform_pca(hidden_tensor[corridor_tensor==i], p)[:, :C.corridor_dim], loc_y[corridor==i]) for i, p in enumerate(pca_corridors)])
    results['res_explained_variance_1st_pc_l'] = np.mean([p['explained_variance_ratio'][0].item() for i, p in enumerate(pca_corridors)])
    results['res_PR'] = (np.sum(pca['explained_variance']) ** 2 / np.sum(pca['explained_variance'] ** 2)).item()
    results['res_PR_l'] = np.mean([(np.sum(p['explained_variance']) ** 2 / np.sum(p['explained_variance'] ** 2)).item() for i, p in enumerate(pca_corridors)])

    hidden_centers = torch.stack([hidden_states[-1][y[:,i]==1].mean(0) for i in range(y.shape[1])])
    hidden_between_cluster_dists = torch.diag(torch.cdist(hidden_centers, hidden_centers),1).mean().item()
    hidden_within_cluster_dists = torch.tensor([get_upper_triangle(torch.cdist(hidden_states[-1][y[:,i]==1], hidden_states[-1][y[:,i]==1])).mean() for i in range(y.shape[1])]).mean().item()
    hidden_mean_norm = torch.linalg.norm(hidden_states[-1], dim=1).mean().item()
    results['res_cluster_collapse_norm_between'] = hidden_within_cluster_dists / hidden_between_cluster_dists
    results['res_cluster_collapse_norm_size'] = hidden_within_cluster_dists / hidden_mean_norm

    results['res_NC1'] = calc_NC1(hidden_tensor.cpu().numpy(), y.cpu().numpy().argmax(1))

    # Create a dictionary with the config values and results
    results = {
            **config_data,
            **C.__dict__.copy(),
            **{'loss': loss_l[-1], 'res_accuracy': accuracy_l[-1]},
            **results,
    }


    # Convert the dictionary to a pandas DataFrame
    results = pd.DataFrame([results])
    return results

    # return data_dict



def worker_function(args):
    config_data = args[0]
    run_name = args[1]
    num_runs = args[2]
    start_time = args[3]
    data_dict_l = args[4]
    result = run_scenario(config_data)
    with counter_lock:
        counter.value += 1
        elapsed_time = (time.time() - start_time) / 60
        total_time = elapsed_time * (num_runs / counter.value)
        time_per_iter = elapsed_time / counter.value
        print(f"Completed {counter.value}/{num_runs} runs --- running time {elapsed_time:.2f} / {total_time:.2f} minutes --- {time_per_iter:.2f} min/iter")

    if result is not None:
        file_name = f'{result_path}/{run_name}.csv'
        if not os.path.isfile(file_name):
            result.to_csv(file_name, index=False, mode='w', header=True)
        else:
            result.to_csv(file_name, index=False, mode='a', header=False)

        # data_dict_l.append(result)
        # if len(data_dict_l) % 10 == 0:
        #     data_dict_l_save = list(data_dict_l)
        #     with open('data_dict_l_from_scan.pkl', 'wb') as f:
                # pkl.dump(data_dict_l_save, f)


def get_args_list_single_var():
    args_list = []
    for var_name in modify_vars.keys():
        iter_items = list(itertools.product(np.arange(0, num_seeds), modify_vars[var_name]))
        for i, (seed, var_val) in enumerate(iter_items):
            # lr = 0.1**(8/2) if var_name != 'L' else 0.1**(var_val/2)
            args_list.append([{**base_params, **{'seed': seed, 'gpu_id': gpu_ids[i % len(gpu_ids)]}, **{var_name: var_val}},
                              f'sweep_params_{var_name}'])
            # args_list.append([{**base_params, **{'seed': seed, 'gpu_id': gpu_ids[i % len(gpu_ids)]}, **{var_name: var_val}},
                            #   f'sweep_params_combo'])
    return args_list

def get_args_list_all_combs():
    args_list = []
    iter_dicts = get_all_key_combinations(modify_vars)
    for i, curr_dict in enumerate(iter_dicts):
        args_list.append([{**base_params, **{'seed': 0, 'gpu_id': gpu_ids[i % len(gpu_ids)], **curr_dict}},
                            f'all_combs'])
    return args_list

if __name__ == "__main__":
    # Your main function code here
    start_time = time.time()
    print('Started parameter sweep.....')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if run_type == 'single_var':
        args_list = get_args_list_single_var()
    elif run_type == 'all_combs':
        args_list = get_args_list_all_combs()
    
    # Create a managed list that can be shared between processes
    manager = Manager()
    data_dict_l = manager.list()
    
    np.random.shuffle(args_list)
    args_list = [a + [len(args_list), start_time, data_dict_l] for a in args_list]
    # subprocess
    counter = mp.Value('i', 0)
    counter_lock = mp.Lock()
    
    if debug:
        for args in args_list:
            worker_function(args)
    else:
        with mp.Pool(processes=num_workers) as pool:
            pool.map(worker_function, args_list) 
    
    # Convert managed list back to regular list before saving
    data_dict_l = list(data_dict_l)
    with open('data_dict_l_from_scan.pkl', 'wb') as f:
        pkl.dump(data_dict_l, f)

    print(f'Total run time: {(time.time() - start_time)/60:.2f} minutes')
