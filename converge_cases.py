from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
from copy import deepcopy

import torch
from torch import nn

from run_sim import Config, run_sim
from utils import cosine_similarity, get_r_2, vector_angle
from tqdm import tqdm
from utils import alignment_score, calc_PR, calc_NC1
import matplotlib as mpl
from functools import reduce

from utils_plot import plot_loss_and_dist, plot_pca
from run_sim import run_sim_wrapper

C = Config()

C.sig_h_2 = 1e-1
C.linear_net = True
C.learning_rate = 0.01
C.L=8
C.print_progress = True
C.early_stopping = True
C.length_corridors = [10]*1
C.max_move = 5
C.hidden_size = 25
C.num_epochs = 10000
C.algo_name = 'SGD'
C.loss_fn = nn.CrossEntropyLoss()

data_dict = run_sim_wrapper(C)

plot_loss_and_dist(data_dict)
plot_pca(data_dict)

torch.save(data_dict['final_weights'], 'model_state_dict.pth')

C.state_dict_path = 'model_state_dict.pth'
C.early_stopping = False
data_dict = run_sim_wrapper(C)

plot_loss_and_dist(data_dict)
plot_pca(data_dict)