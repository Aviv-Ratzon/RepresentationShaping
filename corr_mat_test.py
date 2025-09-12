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
C.length_corridors = [100]*1
for max_move in np.arange(1, C.length_corridors[0]//2-1):
    C.max_move = max_move
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    Sigma31 = X.T@y
    Sigma11 = X.T@X
    Sigma = np.linalg.pinv(Sigma11)@Sigma31
    PR_S_l.append(calc_PR(Sigma[:100]))
    PR_in_l.append(calc_PR(np.linalg.pinv(Sigma11)))
    PR_in_out_l.append(calc_PR(Sigma31))
plt.plot(PR_in_l, marker='o', label='PR_in')
plt.plot(PR_in_out_l, marker='o', label='PR_in_out')
plt.plot(PR_S_l, marker='o', label='PR_S')
plt.yscale('log')
plt.legend()
plt.show()


C.length_corridors = [10]*1
C.max_move = 1* C.length_corridors[0]//2
C.cyclic_corridors = True
X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
Sigma11 = X.T@X
Sigma31 = X.T@y
Sigma = np.linalg.pinv(Sigma11)@Sigma31
U, S, V = np.linalg.svd(Sigma)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(U[:,:2])
axs[1].plot(S, marker='o')
plt.show()
eigs, eigvs = np.linalg.eig(np.linalg.pinv(Sigma11))
plt.plot(eigvs[:,0])
plt.show()

eigs, eigvs = np.linalg.eig(Sigma[:100,:100].astype(np.float64))
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(eigvs[:,:3])
axs[1].plot(eigs, marker='o')
plt.show()

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
im0 = axs[0].imshow(Sigma11)
plt.colorbar(im0, ax=axs[0])
axs[0].set_title('Sigma11')
im1 = axs[1].imshow(np.linalg.pinv(Sigma11))
plt.colorbar(im1, ax=axs[1])
axs[1].set_title('np.linalg.pinv(Sigma11)')
im2 = axs[2].imshow(Sigma31)
plt.colorbar(im2, ax=axs[2])
axs[2].set_title('Sigma31')
im3 = axs[3].imshow(Sigma)
plt.colorbar(im3, ax=axs[3])
axs[3].set_title('Sigma')
plt.tight_layout()
plt.show()

mat = np.array([[1,1,1,1],[1,3,0,0],[1,0,3,0],[1,0,0,3]])
eigs, eigsv = np.linalg.eig(mat)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(eigsv)
axs[1].plot(eigs, marker='o')
plt.show()



n = C.length_corridors[0]
A = n//2-1 # C.max_move
PR_l = []
lamb_l_l = []
for A in range(1,n//2-1):
    c_l = np.ones([n])
    c_l[A+1:-A] = 0
    v_l = []
    lamb_l = []
    for j in range(n):
        w = np.exp(2*np.pi*1j/n)
        vj = w**(j*np.arange(n))/np.sqrt(n)
        lambj = np.sum(c_l*w**(-np.arange(n)*j))
        v_l.append(vj)
        lamb_l.append(lambj)
    v_l = np.array(v_l)
    lamb_l = np.array(lamb_l)
    PR_l.append((lamb_l).sum()**2/(lamb_l**2).sum())
    lamb_l_l.append(lamb_l)
# plt.plot(v_l[:,:3])
# plt.show()
lamb_l_l = np.array(lamb_l_l)
plt.plot(PR_l)
plt.yscale('log')
plt.show()
plt.plot(lamb_l_l[:,2])
plt.show()

plt.imshow(Sigma@Sigma.T)
plt.show()


plt.imshow(Sigma31.T@Sigma31)
plt.show()

plt.plot(S, marker='o')
plt.yscale('log')
plt.show()

L=10
m = 2*L//2+1
A = m*np.eye(L)
B = np.ones((L,m))
C = B.T
D = L*np.eye(m)
XTX = np.block([[A, B], [C, D]])
np.linalg.pinv(XTX)
U, S, V = np.linalg.svd(XTX)
eigs, eigvs = np.linalg.eig(XTX)
plt.plot(U[:,0])
plt.show()
plt.plot(S)
plt.show()
plt.plot(V[:,0])
plt.show()
W_rec = U@np.diag(S)@V.T


C = Config()
C.cyclic_corridors = False
C.length_corridors = [100]*1
Sigma_l = []
X_l = []
y_l = []
action_taken_l = []
max_move_l = [1, C.length_corridors[0]//4, C.length_corridors[0]//2]
for max_move in max_move_l:
    C.max_move = max_move
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    Sigma31 = X.T@y
    Sigma11 = X.T@X
    Sigma = np.linalg.pinv(Sigma11)@Sigma31
    Sigma_l.append(Sigma)
    X_l.append(X)
    y_l.append(y)
    action_taken_l.append(action_taken)

fig, axs = plt.subplots(len(max_move_l), 4, figsize=(4*4, 3*len(max_move_l)))
for i in range(len(Sigma_l)):
    Sigma = Sigma_l[i]
    X = X_l[i]
    y = y_l[i]
    action_taken = action_taken_l[i]
    U, S, V = np.linalg.svd(Sigma)
    axs[i,0].plot(U[:,:2])
    axs[i,1].plot(S, marker='o')
    filter = abs(action_taken)<=1
    axs[i,2].scatter(X[filter]@U[:,0], X[filter]@U[:,1], c=y.argmax(1)[filter], s=1/max_move_l[i])
    axs[i,2].axis('equal')
    accuracy_l = []
    for n in range(2):
        # Project X onto the first n left singular vectors
        X_proj = X @ U[:, :n+1]
        # Train a linear SVM classifier
        from sklearn.svm import SVC
        clf = SVC(kernel='linear', random_state=0, max_iter=1000)
        # Use the projected data to fit the classifier
        clf.fit(X_proj, y.argmax(1))
        # Predict on the same data
        y_pred = clf.predict(X_proj)
        # Compute accuracy
        accuracy = np.mean(y_pred == y.argmax(1))
        accuracy_l.append(accuracy)
    axs[i,3].plot(accuracy_l, marker='o')
    axs[i,3].set_ylim(0, 1)
    axs[i,0].set_ylabel(f'max_move={max_move_l[i]}')
plt.tight_layout()
plt.show()

plt.imshow(np.linalg.pinv(X)@y)
plt.show()

# Create a circular band matrix with band width m and dimension L
L = 100  # Example dimension, can be changed
m = 10   # Example band width, can be changed

# Create the band matrix
band_matrix = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        # Circular distance
        if min((j - i) % L, (i - j) % L) < m:
            band_matrix[i, j] = 1

# Calculate the Participation Ratio
# Participation Ratio = (sum(eigvals)^2) / sum(eigvals^2)
eigvals = np.linalg.eigvalsh(band_matrix)
pr = (np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)

print(f"Circular band matrix (L={L}, m={m}) Participation Ratio: {pr:.2f}")

L = 30
A = 15
mat = np.zeros((L, L))
for i in range(L):
    mat[i, max(0, i-A):min(L, i+A+1)] = 1
plt.imshow(mat)
U, S, V = np.linalg.svd(mat)
plt.plot(U[:,0])

plt.imshow(Sigma)
U, S, V = np.linalg.svd(X, full_matrices=True)
rank = sum(S > 1e-10)
V = V[:rank,:]
U = U[:,:rank]
S = S[:rank]
plt.imshow(V.T@np.diag(S**-1)@U.T@y)

plt.imshow(y)