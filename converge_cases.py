import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm

from run_sim import Config, run_sim_wrapper
from utils import *
from utils_plot import *
import os
from copy import deepcopy



def multiclass_functional_margin(W, x, y):
    W = W / np.linalg.norm(W)
    scores = x@W  # shape (K,)
    true_score = scores[y]
    max_other_score = np.max(np.delete(scores, y))
    return true_score - max_other_score

C = Config()
C.sig_h_2 = 1e-3
C.linear_net = True
C.learning_rate = 1e-2
C.L=5
C.print_progress = True
C.early_stopping = False
C.length_corridors = [8]*1
C.hidden_size = C.length_corridors[0] * 2 + 1
C.num_epochs = int(100000)
C.algo_name = 'SGD'
C.normalize_theta = 10
C.max_move = C.length_corridors[0]//2

var_name = 'max_move'
var_values = np.linspace(1,  C.length_corridors[0], 10).astype(int)
fig, axs = plt.subplots(1, 4, figsize=(6*4, 3))
fig_pca, axs_pca = plt.subplots(len(var_values), 4, figsize=(3*4, 2*len(var_values)))
data_dict_l = []
for var_value, ax_pca in zip(var_values, axs_pca):
    C = deepcopy(C)
    setattr(C, var_name, var_value)
    data_dict = run_sim_wrapper(C)
    data_dict_l.append(data_dict)
    # plot_pca(data_dict, title="Original")

    from copy import deepcopy
    cos_sim_l = []
    order_l = []
    nc1_l = []
    margins_l = []
    for state_dict in tqdm(data_dict['state_dict_l']):
        data_dict_new = deepcopy(data_dict)
        data_dict_new['final_weights'] = state_dict
        W_hidden = get_effective_W_from_model_dict(state_dict, to_hidden=True)
        data_dict_new['hidden_states'] = [data_dict['X'] @ W_hidden]
        theta, _, _ = state_dict_to_theta(state_dict)
        grad = np.mean(compute_gradient_np(data_dict_new, normalize=None, flatten_grads=True),0)
        cos_sim_l.append(cosine_similarity(-grad, theta.cpu().numpy()))
        order_l.append(get_order(data_dict_new))
        nc1_l.append(calc_NC1_from_data_dict(data_dict_new))
        W = get_effective_W_from_model_dict(state_dict).cpu().numpy()
        X = data_dict_new['X'].cpu().numpy()
        y = data_dict_new['y'].cpu().numpy().argmax(1)
        margins_l.append(np.mean([multiclass_functional_margin(W, x_curr, y_curr) for x_curr, y_curr in zip(X, y)]))
    axs[0].plot(cos_sim_l, label=f'{var_name}={var_value}')
    axs[1].plot(order_l, label=f'{var_name}={var_value}')
    axs[2].plot(nc1_l, label=f'{var_name}={var_value}')
    axs[3].plot(margins_l, label=f'{var_name}={var_value}')
    plot_pca(data_dict, title=f'{var_name}={var_value}', axs=ax_pca)
axs[0].set_title('Cosine Similarity of gradient and theta')
axs[1].set_title('Order')
axs[2].set_title('NC1')
axs[3].set_title('Margins')
[ax.set_ylim(-0.2, 1.1) for ax in axs[:-2]]
axs[2].set_yscale('log')
axs[1].legend()
plt.show()


n_dicts = len(data_dict_l)
cos_sim_l = []
order_l = []
nc1_l = []
margins_l = []
margins_synthetic_l = []
for data_dict in data_dict_l:
    state_dict = data_dict['final_weights']
    W_hidden = get_effective_W_from_model_dict(state_dict, to_hidden=True)
    theta, _, _ = state_dict_to_theta(state_dict)
    grad = np.mean(compute_gradient_np(data_dict, normalize=None, flatten_grads=True),0)
    cos_sim_l.append(cosine_similarity(-grad, theta.cpu().numpy()))
    order_l.append(get_order(data_dict))
    nc1_l.append(calc_NC1_from_data_dict(data_dict))
    W = get_effective_W_from_model_dict(state_dict).cpu().numpy()
    X = data_dict['X'].cpu().numpy()
    y = data_dict['y'].cpu().numpy().argmax(1)
    margins_l.append(np.mean([multiclass_functional_margin(W, x_curr, y_curr) for x_curr, y_curr in zip(X, y)]))
    model_dict_synthetic = make_synthetic_model_dict(data_dict)
    W = get_effective_W_from_model_dict(model_dict_synthetic).cpu().numpy()
    margins_synthetic_l.append(np.mean([multiclass_functional_margin(W, x_curr, y_curr) for x_curr, y_curr in zip(X, y)]))
fig, axs = plt.subplots(1, 4, figsize=(6*4, 3))
axs[0].plot(var_values, cos_sim_l, marker='o')
axs[0].set_title('Cosine Similarity of gradient and theta')
axs[1].plot(var_values, order_l, marker='o')
axs[1].set_title('Order')   
axs[2].plot(var_values, nc1_l, marker='o')
axs[2].set_title('NC1'); axs[2].set_yscale('log')
axs[3].plot(var_values, margins_l, marker='o', label='original')
axs[3].plot(var_values, margins_synthetic_l, marker='o', label='synthetic')
axs[3].set_title('Margins')
axs[3].legend()
plt.show()

labels_l = [str(A) for A in var_values]
plot_solution_direction_loss_space(data_dict_l, labels_l)

stop
# model_dict_synthetic = make_synthetic_model_dict(data_dict)
# model_dict_synthetic = normalize_state_dict(model_dict_synthetic, 1)
# C.state_dict_path = 'model_state_dict.pth'
# torch.save(model_dict_synthetic, 'model_state_dict.pth')

# C.early_stopping = False
# C.num_epochs = 2
# data_dict_synthetic = run_sim_wrapper(C)
# plot_pca(data_dict_synthetic, title="Synthetic")

# C.num_epochs = 100000
# C.algo_name = 'SGD'
# C.learning_rate = 1e-4
# C.print_progress = True
# data_dict_converged_syn = run_sim_wrapper(C)
# plot_pca(data_dict_converged_syn, title="Converged synthetic")


# print("Computing Hessians")
# H = compute_hessian(data_dict_synthetic, normalize=None)
# print("Computing Eigenvalues")
# eigs, eigs_v = torch.linalg.eig(H)
# accuracy_map, r_2_map, norm_l, n_eigs_l = grid_search_pert_direction(data_dict_synthetic, eigs, eigs_v)

# filtered_map = r_2_map
# filtered_map[accuracy_map<0.99] = 1
# i_eigs, i_norm = np.unravel_index(np.argmin(filtered_map), filtered_map.shape)
# n_eigs = n_eigs_l[i_eigs]
# norm = norm_l[i_norm]
# new_weights = perturb_model_dict(data_dict_synthetic['final_weights'], torch.real(eigs_v[:, abs(eigs).argsort()[:n_eigs]].mean(axis=1)), norm=norm, normalize=10)
# torch.save(new_weights, 'model_state_dict.pth')
# C.state_dict_path = 'model_state_dict.pth'
# C.num_epochs = 2
# data_dict_perturbed = run_sim_wrapper(C)
# plot_pca(data_dict_perturbed, title="Perturbed synthetic")

# C.num_epochs = 100000
# C.algo_name = 'SGD'
# C.learning_rate = 1e-4
# C.print_progress = True
# data_dict_converged_syn_pert = run_sim_wrapper(C)
# plot_pca(data_dict_converged_syn_pert, title="Converged synthetic perturbed")

# # grad = compute_gradient(data_dict_synthetic, normalize=1)

# # plot_loss_and_dist(data_dict)
# plot_pca(data_dict)
# torch.save(data_dict['final_weights'], 'model_state_dict.pth')

# C.state_dict_path = 'model_state_dict.pth'

# C.early_stopping = False
# data_dict = run_sim_wrapper(C)

# print(f"Accuracy: {(data_dict['outputs'].argmax(1) == data_dict['y'].argmax(1)).float().mean()}")
# # plot_loss_and_dist(data_dict)
# plot_loss_and_dist(data_dict)
# plot_pca(data_dict)

pert_from_norm = None
print("Computing Hessians")
H = compute_hessian(data_dict, normalize=pert_from_norm)
print("Computing Eigenvalues")
eigs, eigs_v = torch.linalg.eig(H)
accuracy_map, r_2_map, norm_l, n_eigs_l = grid_search_pert_direction(data_dict, eigs, eigs_v, pert_from_norm=pert_from_norm, norm_range=(20, 35), n_eigs_range=(1, 750))

filtered_map = r_2_map
filtered_map[accuracy_map<0.99] = 1
plt.imshow(filtered_map)
plt.colorbar()
plt.xticks(np.arange(len(norm_l))[::len(norm_l)//10], norm_l[::len(norm_l)//10].astype(int), rotation=45)
plt.yticks(np.arange(len(n_eigs_l))[::len(n_eigs_l)//10], n_eigs_l[::len(n_eigs_l)//10], rotation=45)
plt.xlabel('norm')
plt.ylabel('n_eigs')
plt.show()

i_eigs, i_norm = np.unravel_index(np.argmin(filtered_map), filtered_map.shape)
n_eigs = n_eigs_l[i_eigs]
norm = norm_l[i_norm]

new_weights = perturb_model_dict(data_dict['final_weights'], torch.real(eigs_v[:, abs(eigs).argsort()[:n_eigs]].mean(axis=1)), norm=norm, normalize=14)
torch.save(new_weights, 'model_state_dict.pth')
C.state_dict_path = 'model_state_dict.pth'
C.num_epochs = 1
data_dict_perturbed = run_sim_wrapper(C)

print(f"Accuracy of perturbed: {(data_dict_perturbed['outputs'].argmax(1) == data_dict_perturbed['y'].argmax(1)).float().mean()}")
plot_pca(data_dict_perturbed)

C.num_epochs = 100000
C.print_progress = True
data_dict_converged = run_sim_wrapper(C)
plot_pca(data_dict_converged)



# print("Computing Hessians")
# H_perturbed = compute_hessian(data_dict_perturbed, normalize=1)
# H_converged = compute_hessian(data_dict_converged, normalize=1)
# print("Computing Eigenvalues")
# eigs_perturbed, eigs_v_perturbed = torch.linalg.eig(H_perturbed)
# eigs_converged, eigs_v_converged = torch.linalg.eig(H_converged)

# plt.plot(abs(eigs_converged.cpu().numpy()), label='converged')
# plt.plot(abs(eigs_perturbed.cpu().numpy()), label='perturbed')
# plt.legend()
# plt.yscale('log')
# plt.show()

# data_dict_l = [data_dict, data_dict_perturbed, data_dict_synthetic, data_dict_converged]
# labels_l = ['original', 'perturbed', 'synthetic', 'converged']
# plot_solution_direction_loss_space(data_dict_l, labels_l)

# n_params = H.shape[0]
# new_weights = perturb_model_dict(data_dict['final_weights'], torch.randn(size=[n_params]), norm=1000, normalize=1)
# torch.save(new_weights, 'model_state_dict.pth')
# C.s
# data_dict_random = run_sim_wrapper(C)

data_dict_l = [data_dict]
labels_l = ['original']
labels_l = [str(A) for A in max_move_l]
plot_solution_direction_loss_space(data_dict_l, labels_l)

data_dict_l = [data_dict, data_dict_perturbed, data_dict_converged]
labels_l = ['original', 'perturbed', 'converged']
plot_solution_direction_loss_space(data_dict_l, labels_l)

# H_random = compute_hessian(data_dict_random, normalize=10)
# H = compute_hessian(data_dict, normalize=10)
# eigs_random, eigs_v_random = torch.linalg.eig(H_random)
# eigs, eigs_v = torch.linalg.eig(H)
# eigs_random = eigs_random.cpu().numpy()
# eigs = eigs.cpu().numpy()

# # plt.plot(eigs_random, label='random', lw=1, alpha=0.7)
# plt.plot(eigs, label='original', ls='--', alpha=0.7)
# plt.legend()
# plt.xscale('log')
# plt.show()



# grads = compute_gradient_np(data_dict, normalize=1)
# grads_perturbed = compute_gradient_np(data_dict_perturbed, normalize=1)

# print(np.linalg.norm(np.mean(grads, axis=0)), np.linalg.norm(np.mean(grads_perturbed, axis=0)))
# plt.plot(np.linalg.norm(grads, axis=1))
# plt.gca().twinx().plot(np.linalg.norm(grads_perturbed, axis=1), color='red')
# # plt.axis('equal')
# plt.show()


# from functools import reduce

# L_l = np.linspace(2,100,10).astype(int)
# N_l = np.linspace(2,1000, 10).astype(int)
# PR_l = np.zeros((len(L_l), len(N_l)))
# for i, L in tqdm(enumerate(L_l)):
#     for j, N in enumerate(N_l):
#         W_l = [np.random.randn(N,N)/10 for _ in range(L)]
#         W_effective = reduce(np.matmul, W_l)
#         PR_l[i,j] = calc_PR(W_effective)

# plt.imshow(PR_l)
# plt.xticks(np.arange(len(N_l)), N_l, rotation=45)
# plt.yticks(np.arange(len(L_l)), L_l, rotation=45)
# plt.xlabel('N')
# plt.ylabel('L')
# plt.colorbar()
# plt.show()


# Ux, Sx, Vx = np.linalg.svd(data_dict['X'].cpu().numpy())
# W1 = data_dict['final_weights']['input_layer.weight'].cpu().numpy()
# Uw, Sw, Vw = np.linalg.svd(W1)
# plt.plot(Sx)
# plt.plot(Sw)
# plt.show()

# plt.scatter(Vx[:, 0], Uw[0, :])

norm = get_state_dict_norm(data_dict['initial_weights']).item()
grads_l = compute_gradient_np(data_dict, normalize=norm, flatten_grads=False)
grad_norm_l = [np.linalg.norm(g) for g in flatten_list(grads_l)]
grads_l_perturbed = compute_gradient_np(data_dict_perturbed, normalize=norm, flatten_grads=False)
grad_norm_l_perturbed = [np.linalg.norm(g) for g in flatten_list(grads_l_perturbed)]

plt.figure(figsize=(15, 5))
plt.plot(grad_norm_l)
plt.plot(grad_norm_l_perturbed)
plt.title(f'||g|| = {sum(grad_norm_l)} --- ||g_perturbed|| = {sum(grad_norm_l_perturbed)}')
plt.show()



print(get_state_dict_norm(data_dict['initial_weights']))
print(get_state_dict_norm(data_dict['final_weights']))
print(get_state_dict_norm(data_dict_perturbed['final_weights']))


W_l = [W.clone().detach() for W in data_dict['final_weights'].values()]
W_l = normalize_W_l(W_l, norm)
W_l_perturbed = [W.clone().detach() for W in data_dict_perturbed['final_weights'].values()]
W_l_perturbed = normalize_W_l(W_l_perturbed, norm)

plt.plot([np.linalg.norm(W)**2 for W in W_l])
plt.plot([np.linalg.norm(W)**2 for W in W_l_perturbed])
plt.show()


new_weights = perturb_model_dict(data_dict['final_weights'], torch.real(eigs_v[:, abs(eigs).argsort()[:n_eigs]].mean(axis=1)), norm=norm, normalize=5)
torch.save(new_weights, 'model_state_dict.pth')
C.state_dict_path = 'model_state_dict.pth'
C.num_epochs = 100000
data_dict_converged = run_sim_wrapper(C)
plot_pca(data_dict_converged)
data_dict_l = [data_dict, data_dict_converged, data_dict_perturbed]
labels_l = ['original', 'converged', 'perturbed']
plot_solution_direction_loss_space(data_dict_l, labels_l)



new_weights = normalize_state_dict(data_dict['final_weights'], 7.5)
torch.save(new_weights, 'model_state_dict.pth')
C.state_dict_path = 'model_state_dict.pth'
C.num_epochs = 1000
data_dict_from_small = run_sim_wrapper(C)
plot_pca(data_dict_from_small)

initial_theta, _, _ = state_dict_to_theta(data_dict['initial_weights'])
final_theta, _, _ = state_dict_to_theta(data_dict['final_weights'])
from_small_theta, _, _ = state_dict_to_theta(data_dict_from_small['final_weights'])
perturbed_theta, _, _ = state_dict_to_theta(data_dict_perturbed['final_weights'])

print(f"cosine similarity of initial and final: {cosine_similarity(initial_theta, final_theta)}")
print(f"cosine similarity of initial and perturbed: {cosine_similarity(initial_theta, perturbed_theta)}")
print(f"cosine similarity of initial and from small: {cosine_similarity(initial_theta, from_small_theta)}")
print(f"cosine similarity of final and from small: {cosine_similarity(final_theta, from_small_theta)}")

cos_sim_l = []
for state_dict in tqdm(data_dict_converged['state_dict_l']):
    data_dict_new = deepcopy(data_dict_converged)
    data_dict_new['final_weights'] = state_dict
    theta, _, _ = state_dict_to_theta(state_dict)
    grad = np.mean(compute_gradient_np(data_dict_new, normalize=None, flatten_grads=True),0)
    cos_sim_l.append(cosine_similarity(grad, final_theta.cpu().numpy()))
plt.plot(cos_sim_l)
plt.show()



data_dict_l = [data_dict]
labels_l = ['original']
plot_solution_direction_loss_space(data_dict_l, labels_l)

from copy import deepcopy
state_dict = deepcopy(data_dict['final_weights'])
state_dict = normalize_state_dict(state_dict, 1)
norms = np.linspace(0.1, 10, 1000)
resulting_norms_l = []
for norm in norms:
    state_dict_new = {k: v*np.sqrt(norm) for k, v in state_dict.items()}
    resulting_norms_l.append(get_state_dict_norm(state_dict_new))
plt.plot(norms, resulting_norms_l)
plt.show()

norm_l = np.linspace(0.1, 12, 1000)
delta_norm_l = []
h_norm_l = []
for norm in norm_l:
    delta_l_samples, h_l_samples = get_delta_and_h(data_dict, normalize=norm)
    delta_norm_l.append(np.linalg.norm(np.concatenate([np.mean(delta,0) for delta in delta_l_samples])))
    h_norm_l.append(np.linalg.norm(np.concatenate([np.mean(h,0) for h in h_l_samples])))
plt.plot(norm_l, delta_norm_l, label='delta')
plt.plot(norm_l, h_norm_l, label='h')
plt.plot(norm_l, np.array(h_norm_l)*np.array(delta_norm_l), label='h*delta')
plt.legend()
plt.yscale('log')
plt.show()

v1 = np.random.normal(size=[1000,1])
v2 = np.random.normal(size=[1,1000])
print(np.linalg.norm(v1))
print(np.linalg.norm(v2))
print(np.linalg.norm(0.5*np.array(v1)+0.5*np.array(v2)))


norm = 5
print(get_loss(data_dict, norm))
print(get_loss(data_dict_perturbed, norm))

W_norm = torch.linalg.norm(get_effective_W_from_model_dict(data_dict['final_weights'], normalize=norm)).item()
W_norm_perturbed = torch.linalg.norm(get_effective_W_from_model_dict(data_dict_perturbed['final_weights'], normalize=norm)).item()

print(W_norm)
print(W_norm_perturbed)







