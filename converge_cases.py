from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from utils_plot import plot_loss_and_dist, plot_pca
from run_sim import run_sim_wrapper, Config
from utils import compute_hessian, get_r_2, normalize_state_dict, perturb_model_dict, get_loss, make_synthetic_model_dict
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
C.hidden_size = 21
C.num_epochs = 100000
C.algo_name = 'SGD'
C.loss_fn = nn.CrossEntropyLoss()

data_dict = run_sim_wrapper(C)

# plot_loss_and_dist(data_dict)
data_dict['loc_y'] = data_dict['loc_y'][:, 0]
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
H = compute_hessian(data_dict)
print("Computing Eigenvalues")
eigs, eigs_v = torch.linalg.eig(H)

import os
C.print_progress = False
n_samples = 100
accuracy_map = np.zeros((n_samples, n_samples))
r_2_map = np.zeros((n_samples, n_samples))
n_eigs_l = np.linspace(1, len(eigs)-1, n_samples).astype(int)
norm_l = np.linspace(0.1, 100, n_samples)
for i, n_eigs in tqdm(enumerate(n_eigs_l)):
    for j, norm in enumerate(norm_l):
        new_weights = perturb_model_dict(data_dict['final_weights'], torch.real(eigs_v[:, abs(eigs).argsort()[:n_eigs]].mean(axis=1)), norm=norm, normalize=1)
        torch.save(new_weights, 'model_state_dict.pth')

        C.num_epochs = 0
        data_dict_perturbed = run_sim_wrapper(C)
        h_np = data_dict_perturbed['hidden_states'][-1].cpu().detach().numpy()
        accuracy = (data_dict_perturbed['outputs'].argmax(1) == data_dict_perturbed['y'].argmax(1)).float().mean()
        r_2 = get_r_2(PCA(n_components=2).fit_transform(h_np), data_dict_perturbed['loc_y'])
        accuracy_map[i, j] = accuracy
        r_2_map[i, j] = r_2
        os.remove('model_state_dict.pth')
        if accuracy >=0.99 and r_2 < 0.5:
            print(f"Accuracy of perturbed: {accuracy}, R^2: {r_2}, Norm: {norm}, N_eigs: {n_eigs}")
            break

map = r_2_map
map[accuracy_map<0.99] = 1
plt.imshow(map)
plt.colorbar()
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

# W_l_synthetic = make_synthetic_model_dict(data_dict, normalize=None)
# new_model_dict = {k:v for k, v in zip(data_dict['final_weights'].keys(), W_l_synthetic)}
# torch.save(new_model_dict, 'model_state_dict.pth')
# C.num_epochs = 0
# data_dict_synthetic = run_sim_wrapper(C)
# plot_pca(data_dict_synthetic)

norm_l = np.linspace(0.1, 50, 1000)
plt.plot(norm_l, [get_loss(data_dict, normalize=norm) for norm in norm_l], label='original')
plt.plot(norm_l, [get_loss(data_dict_perturbed, normalize=norm) for norm in norm_l], label='perturbed')
plt.plot(norm_l, [get_loss(data_dict_converged, normalize=norm) for norm in norm_l], label='converged')
# plt.plot(norm_l, [get_loss(data_dict_synthetic, normalize=norm) for norm in norm_l], label='synthetic')
plt.legend()
plt.yscale('log')
plt.xlabel('Normalization Factor')
plt.ylabel('Loss')
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

def factorize_matrix_to_L_matrices(W, L, N=None):
    """
    Factorize a matrix W into L matrices with intermediate dimension N.
    
    Args:
        W: Input matrix of shape (m, n)
        L: Number of matrices to factorize into
        N: Intermediate dimension. If None, uses min(m, n)
    
    Returns:
        List of L matrices whose multiplication equals W
    """
    m, n = W.shape
    
    if N is None:
        N = min(m, n)
    
    # Initialize matrices randomly
    matrices = []
    
    # First matrix: shape (m, N)
    matrices.append(torch.randn(m, N, requires_grad=True))
    
    # Middle matrices: shape (N, N)
    for i in range(L - 2):
        matrices.append(torch.randn(N, N, requires_grad=True))
    
    # Last matrix: shape (N, n)
    matrices.append(torch.randn(N, n, requires_grad=True))
    print([W.shape for W in matrices])
    
    # Optimize to find the factorization
    optimizer = torch.optim.Adam(matrices, lr=0.00001)
    
    n_steps = 100000
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Compute the product of all matrices
        product = matrices[0]
        for i in range(1, L):
            product = product @ matrices[i]

        # Compute loss (MSE between product and target W)
        loss = torch.nn.functional.mse_loss(product, W)
        
        loss.backward()
        optimizer.step()
        
        if step % (n_steps//10) == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
    
    return matrices

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