import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from multiprocessing import Pool
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
    
    return W, margins, min_margin_idx, data_dict

if __name__ == '__main__':
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
    # This removes the non-serializable loss_fn
    config_dict = {attr: getattr(C, attr) for attr in dir(C) 
                   if not attr.startswith('_') and not callable(getattr(C, attr))}

    # Create process pool
    num_processes = min(len(var_values), mp.cpu_count())
    pool = Pool(processes=num_processes)

    # Run simulations in parallel
    args = [(var_value, var_name, config_dict) for var_value in var_values]
    results = pool.map(process_var_value, args)

    # Close pool
    pool.close()
    pool.join()

    # Unpack results
    W_l, margins_l, min_margin_idx_l, data_dict_l = zip(*results)

    # Plot results
    import matplotlib.pyplot as plt
    plt.plot(var_values, margins_l)
    plt.xlabel(var_name)
    plt.ylabel('Min Margin')
    plt.title(f'Min Margin vs {var_name}')
    plt.show() 