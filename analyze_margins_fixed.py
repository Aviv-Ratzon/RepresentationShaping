import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from multiprocessing import Pool, freeze_support
import multiprocessing as mp
from run_sim import Config, run_sim_wrapper
from utils import get_effective_W_from_model_dict

def multiclass_functional_margin(W, X, y, reducer=np.min):
    W = W / np.linalg.norm(W)
    margins = []
    i_max_other_score_l = []
    for x, y_curr in zip(X, y):
        label = y_curr.argmax()
        scores = x@W  # shape (K,)
        true_score = scores[label]
        max_other_score = np.max(np.delete(scores, label))
        i_max_other_score = np.argmax(np.delete(scores, label))
        margins.append(true_score - max_other_score)
        i_max_other_score_l.append(i_max_other_score)
    return reducer(margins), np.argmin(margins), i_max_other_score_l[np.argmin(margins)]

def process_var_value(args):
    var_value, var_name, config_dict = args
    print(f'Running {var_name} = {var_value}')
    
    # Recreate the Config object from the dictionary
    C = Config()
    for key, value in config_dict.items():
        setattr(C, key, value)
    
    # Set the variable we want to vary
    setattr(C, var_name, var_value)
    
    # Recreate the loss function (it was removed for serialization)
    C.loss_fn = nn.CrossEntropyLoss()
    
    data_dict = run_sim_wrapper(C)
    
    W = get_effective_W_from_model_dict(data_dict['final_weights']).cpu().numpy()
    X = data_dict['X'].cpu().numpy()
    y = data_dict['y'].cpu().numpy()
    action_taken = data_dict['action_taken']
    loc_y = data_dict['loc_y']
    loc_X = data_dict['loc_X']

    margins, min_margin_idx, i_max_other_score = multiclass_functional_margin(W, X, y)
    
    print(f'Min margin sample: loc_X: {loc_X[min_margin_idx]}, action_taken: {action_taken[min_margin_idx]}, loc_y: {loc_y[min_margin_idx]}, i_max_other_score: {i_max_other_score}')
    
    # Return only essential data to reduce serialization overhead
    return {
        'var_value': var_value,
        'W': W,
        'margins': margins,
        'min_margin_idx': min_margin_idx,
        'loc_X_min': loc_X[min_margin_idx],
        'action_taken_min': action_taken[min_margin_idx],
        'loc_y_min': loc_y[min_margin_idx],
        'i_max_other_score': i_max_other_score
    }

def run_parallel_simulations():
    # Create your config
    C = Config()
    C.G = 0.8
    C.sig_h_2 = 1e-3
    C.linear_net = True
    C.learning_rate = 0.001
    C.L = 5
    C.print_progress = True
    C.length_corridors = [5]*1
    C.max_move = C.length_corridors[0]//2
    C.hidden_size = 100
    C.num_epochs = 100
    C.algo_name = 'SGD'
    C.loss_fn = nn.CrossEntropyLoss()

    # Parameters for the sweep
    var_name = 'L'
    var_values = np.linspace(1, 10, 10).astype(int)

    # Create a serializable dictionary from the Config object
    config_dict = {attr: getattr(C, attr) for attr in dir(C) 
                   if not attr.startswith('_') and not callable(getattr(C, attr))}

    # Use fewer processes to reduce memory pressure
    num_processes = min(len(var_values), 4, mp.cpu_count())
    print(f"Using {num_processes} processes")
    
    # Create process pool with proper Windows settings
    pool = Pool(processes=num_processes)

    # Run simulations in parallel
    args = [(var_value, var_name, config_dict) for var_value in var_values]
    results = pool.map(process_var_value, args)

    # Close pool
    pool.close()
    pool.join()

    # Sort results by var_value
    results.sort(key=lambda x: x['var_value'])
    
    # Extract data
    var_values_sorted = [r['var_value'] for r in results]
    W_l = [r['W'] for r in results]
    margins_l = [r['margins'] for r in results]
    min_margin_idx_l = [r['min_margin_idx'] for r in results]

    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(var_values_sorted, margins_l, 'o-', linewidth=2, markersize=8)
    plt.xlabel(var_name, fontsize=12)
    plt.ylabel('Min Margin', fontsize=12)
    plt.title(f'Min Margin vs {var_name}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return W_l, margins_l, min_margin_idx_l, results

if __name__ == '__main__':
    # This is crucial for Windows multiprocessing
    freeze_support()
    W_l, margins_l, min_margin_idx_l, results = run_parallel_simulations()
    print("Simulation completed successfully!") 