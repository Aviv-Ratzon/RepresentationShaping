from utils import *
from data_modules import create_data
from run_sim import Config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

C = Config()
C.cyclic_corridors = False
PR_S_l = []
PR_in_l = []
PR_in_out_l = []
C.length_corridors = [10]*1
for max_move in np.arange(C.length_corridors[0]):
    C.max_move = max_move
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
    Sigma31 = X.T@y
    Sigma11 = X.T@X
    Sigma = np.linalg.pinv(Sigma11)@Sigma31
    PR_S_l.append(calc_PR(Sigma))
    PR_in_l.append(calc_PR(np.linalg.pinv(Sigma11)))
    PR_in_out_l.append(calc_PR(Sigma31))
plt.plot(PR_S_l, marker='o', label='PR_S')
plt.plot(PR_in_l, marker='o', label='PR_in')
plt.plot(PR_in_out_l, marker='o', label='PR_in_out')
plt.legend()
plt.show()


C.max_move = 5
X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
Sigma11 = X.T@X
eigs, eigvs = np.linalg.eig(np.linalg.pinv(Sigma11))
plt.plot(eigvs[:,0])
plt.show()

