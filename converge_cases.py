import torch
from torch import nn

from utils_plot import plot_loss_and_dist, plot_pca
from run_sim import run_sim_wrapper, Config
from utils import compute_hessian
import matplotlib.pyplot as plt

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

H = compute_hessian(data_dict)
eigs, eigs_v = torch.linalg.eigh(H)

plt.plot(sorted(abs(eigs).cpu().numpy()))
plt.yscale('log')
plt.show()

perturb_model_dict(data_dict['final_weights'], eigs_v[abs(eigs).argmin()], norm=1)