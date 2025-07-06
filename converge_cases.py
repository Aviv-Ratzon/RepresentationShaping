import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm

from run_sim import Config, run_sim_wrapper
from utils import (compute_gradient, compute_hessian, get_loss, get_r_2, grid_search_pert_direction,
                   make_synthetic_model_dict, normalize_W_l, normalize_state_dict,
                   perturb_model_dict)
from utils_plot import *
import os

C = Config()

C.sig_h_2 = 1e-1
C.linear_net = True
C.learning_rate = 1e-1
C.L=10
C.print_progress = True
C.early_stopping = False
C.length_corridors = [10]*1
C.max_move = 5
C.hidden_size = 50
C.num_epochs = 10
C.algo_name = 'SGD'
C.loss_fn = nn.CrossEntropyLoss()


data_dict = run_sim_wrapper(C)
plot_pca(data_dict, title="Original")


model_dict_synthetic = make_synthetic_model_dict(data_dict)
model_dict_synthetic = normalize_state_dict(model_dict_synthetic, 1)
C.state_dict_path = 'model_state_dict.pth'
torch.save(model_dict_synthetic, 'model_state_dict.pth')

C.early_stopping = False
C.num_epochs = 2
data_dict_synthetic = run_sim_wrapper(C)
plot_pca(data_dict_synthetic, title="Synthetic")

C.num_epochs = 100000
C.algo_name = 'SGD'
C.learning_rate = 1e-4
C.print_progress = True
data_dict_converged_syn = run_sim_wrapper(C)
plot_pca(data_dict_converged_syn, title="Converged synthetic")


print("Computing Hessians")
H = compute_hessian(data_dict_synthetic, normalize=None)
print("Computing Eigenvalues")
eigs, eigs_v = torch.linalg.eig(H)
accuracy_map, r_2_map, norm_l, n_eigs_l = grid_search_pert_direction(data_dict_synthetic, eigs, eigs_v)

filtered_map = r_2_map
filtered_map[accuracy_map<0.99] = 1
i_eigs, i_norm = np.unravel_index(np.argmin(filtered_map), filtered_map.shape)
n_eigs = n_eigs_l[i_eigs]
norm = norm_l[i_norm]
new_weights = perturb_model_dict(data_dict_synthetic['final_weights'], torch.real(eigs_v[:, abs(eigs).argsort()[:n_eigs]].mean(axis=1)), norm=norm, normalize=10)
torch.save(new_weights, 'model_state_dict.pth')
C.state_dict_path = 'model_state_dict.pth'
C.num_epochs = 2
data_dict_perturbed = run_sim_wrapper(C)
plot_pca(data_dict_perturbed, title="Perturbed synthetic")

C.num_epochs = 100000
C.algo_name = 'SGD'
C.learning_rate = 1e-4
C.print_progress = True
data_dict_converged_syn_pert = run_sim_wrapper(C)
plot_pca(data_dict_converged_syn_pert, title="Converged synthetic perturbed")

# grad = compute_gradient(data_dict_synthetic, normalize=1)

# plot_loss_and_dist(data_dict)
plot_pca(data_dict)
torch.save(data_dict['final_weights'], 'model_state_dict.pth')

C.state_dict_path = 'model_state_dict.pth'

C.early_stopping = False
data_dict = run_sim_wrapper(C)

print(f"Accuracy: {(data_dict['outputs'].argmax(1) == data_dict['y'].argmax(1)).float().mean()}")
# plot_loss_and_dist(data_dict)
plot_loss_and_dist(data_dict)
plot_pca(data_dict)

print("Computing Hessians")
H = compute_hessian(data_dict, normalize=None)
print("Computing Eigenvalues")
eigs, eigs_v = torch.linalg.eig(H)
accuracy_map, r_2_map, norm_l, n_eigs_l, eigs, eigs_v = grid_search_pert_direction(data_dict)

map = r_2_map
map[accuracy_map<0.99] = 1
plt.imshow(filtered_map)
plt.colorbar()
plt.xticks(np.arange(len(norm_l))[::len(norm_l)//10], norm_l[::len(norm_l)//10].astype(int), rotation=45)
plt.yticks(np.arange(len(n_eigs_l))[::len(n_eigs_l)//10], n_eigs_l[::len(n_eigs_l)//10], rotation=45)
plt.xlabel('norm')
plt.ylabel('n_eigs')
plt.show()
i_eigs, i_norm = np.unravel_index(np.argmin(map), map.shape)
n_eigs = n_eigs_l[i_eigs]
norm = norm_l[i_norm]

new_weights = perturb_model_dict(data_dict['final_weights'], torch.real(eigs_v[:, abs(eigs).argsort()[:n_eigs]].mean(axis=1)), norm=norm, normalize=10)
torch.save(new_weights, 'model_state_dict.pth')
C.num_epochs = 0
data_dict_perturbed = run_sim_wrapper(C)

print(f"Accuracy of perturbed: {(data_dict_perturbed['outputs'].argmax(1) == data_dict_perturbed['y'].argmax(1)).float().mean()}")
plot_pca(data_dict_perturbed)

C.num_epochs = 10000
C.print_progress = True
data_dict_converged = run_sim_wrapper(C)

plot_loss_and_dist(data_dict_converged)
plot_pca(data_dict_converged)

print(f"Accuracy of converged: {(data_dict_converged['outputs'].argmax(1) == data_dict_converged['y'].argmax(1)).float().mean()}")


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

data_dict_l = [data_dict, data_dict_perturbed, data_dict_synthetic, data_dict_converged]
labels_l = ['original', 'perturbed', 'synthetic', 'converged']
plot_solution_direction_loss_space(data_dict_l, labels_l)

n_params = H.shape[0]
new_weights = perturb_model_dict(data_dict['final_weights'], torch.randn(size=[n_params]), norm=1000, normalize=1)
torch.save(new_weights, 'model_state_dict.pth')
C.s
data_dict_random = run_sim_wrapper(C)

data_dict_l = [data_dict_synthetic, data_dict_converged]
labels_l = ['synthetic', 'converged']
plot_solution_direction_loss_space(data_dict_l, labels_l)

H_random = compute_hessian(data_dict_random, normalize=10)
H = compute_hessian(data_dict, normalize=10)
eigs_random, eigs_v_random = torch.linalg.eig(H_random)
eigs, eigs_v = torch.linalg.eig(H)
eigs_random = eigs_random.cpu().numpy()
eigs = eigs.cpu().numpy()

# plt.plot(eigs_random, label='random', lw=1, alpha=0.7)
plt.plot(eigs, label='original', ls='--', alpha=0.7)
plt.legend()
plt.xscale('log')
plt.show()

X_np = data_dict['X'].cpu().numpy()
y_np = data_dict['y'].cpu().numpy()
W_logistic_noreg = LogisticRegression(multi_class='multinomial', penalty=None, fit_intercept=False, tol=1e-12).fit(X_np, y_np.argmax(1)).coef_
print(f"Accuracy of logistic regression: {accuracy_score((X_np@W_logistic_noreg.T).argmax(1), y_np.argmax(1))}")

# # Add SVM solution
# from sklearn.svm import LinearSVC
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# svm_model = LinearSVC(tol=1e-12, C=10000, max_iter=1000000, fit_intercept=False, multi_class='crammer_singer').fit(X_np, y_np.argmax(1))
# svm_accuracy = svm_model.score(X_np, y_np.argmax(1))
# print(f"Accuracy of SVM: {svm_accuracy}")

# # Extract the linear classifier coefficients
# W_logistic_noreg = svm_model.coef_  # Shape: (n_classes, n_features)


# Verify it matches the sklearn score
print(f"Manual accuracy matches sklearn score: {abs(svm_accuracy - svm_manual_accuracy) < 1e-10}")


W_logistic = torch.tensor(W_logistic_noreg).float()
W_l_logsitic_factorized = factorize_matrix_to_L_matrices(W_logistic.T, C.L+1, C.hidden_size)
W_l_logsitic_factorized = [W.T for W in W_l_logsitic_factorized]
W_logistic_factorized = W_l_logsitic_factorized[0].T
for W in W_l_logsitic_factorized[1:]:
    W_logistic_factorized = W_logistic_factorized @ W.T
print([w.shape for w in W_l_logsitic_factorized])
print(W_logistic_factorized.shape, W_logistic.shape)
X = data_dict['X']; y = data_dict['y']
print(f"MSE between W and W_l_final: {torch.nn.functional.mse_loss(W_logistic_factorized.T, W_logistic).item():.6f}")
print(f"Accuracy of logistic regression factorized: {accuracy_score((X@W_logistic_factorized).argmax(1), y.argmax(1))}")

new_model_dict = {k:v.clone().detach() for k, v in zip(data_dict['final_weights'].keys(), W_l_logsitic_factorized)}
new_model_dict = normalize_state_dict(new_model_dict, norm=0.1)
torch.save(new_model_dict, 'model_state_dict.pth')

C.num_epochs = 0
data_dict_factorized = run_sim_wrapper(C)
plot_pca(data_dict_factorized)
print(f"Accuracy of factorized: {(data_dict_factorized['outputs'].argmax(1) == data_dict_factorized['y'].argmax(1)).float().mean()}")


C.num_epochs = 10000
data_dict_converged_2 = run_sim_wrapper(C)
print(f"Accuracy of converged_2: {(data_dict_converged_2['outputs'].argmax(1) == data_dict_converged_2['y'].argmax(1)).float().mean()}")
plot_loss_and_dist(data_dict_converged_2)
plot_pca(data_dict_converged_2)

norm_l = np.linspace(0.1, 300, 1000)
plt.plot(norm_l, [get_loss(data_dict, normalize=norm) for norm in norm_l], label='original')
plt.plot(norm_l, [get_loss(data_dict_factorized, normalize=norm) for norm in norm_l], label='logistic factorized')
plt.plot(norm_l, [get_loss(data_dict_converged_2, normalize=norm) for norm in norm_l], label='converged_2')
# plt.plot(norm_l, [get_loss(data_dict_synthetic, normalize=norm) for norm in norm_l], label='synthetic')
plt.legend()
plt.yscale('log')
plt.xlabel('Normalization Factor')
plt.ylabel('Loss')
plt.show()

print("Computing Hessian")
H_converged_2 = compute_hessian(data_dict_converged_2, normalize=1)
print("Computing Eigenvalues")
eigs_converged_2, eigs_v_converged_2 = torch.linalg.eig(H_converged_2)