import itertools
import os
import multiprocessing as mp
import time
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
import numpy as np
from torch.nn import MSELoss

from run_sim_hyper import Config, run_sim
from utils import cosine_similarity, get_r_2, vector_angle, get_upper_triangle, calc_NC1

C = Config()
C.length_corridors = [3]
C.corridor_dim = 2
C.max_move = 1
C.print_progress = True

X, y, corridor, loc_X, loc_y, action_taken, hidden_states, loss_l, accuracy_l, outputs, hidden_l, final_weights, initial_weights = run_sim(C)

print(loss_l[-1], accuracy_l[-1])