import numpy as np
import matplotlib.pyplot as plt
from run_sim import Config, run_sim, run_sim_wrapper, create_data
from tqdm import tqdm
from utils import get_r_2

overlap_l = []
S_l = np.arange(11, 50, 2) # [21] # 
plot = len(S_l) <= 2
for S in tqdm(S_l):
    # S = 100
    A = S//2 # S//2 # 1
    d1 = 2 * A + 1
    d2 = S
    d = d1 + d2

    M = np.zeros((d, S))
    for s in range(S):
        low = max(0, s-A)
        high = min(S-1, s+A)
        prob = 1 / (high - low + 1)
        M[s, low:high+1] = prob

    for a_i, a in enumerate(range(-A,A+1)):
        low = max(0, a)
        high = min(S-1, S+a-1)
        prob = 1 / (high - low + 1)
        M[d2 + a_i, low:high+1] = prob

    mu = np.zeros(d2)
    for s in range(S):
        low = max(0, s-A)
        high = min(S-1, s+A)
        n = (high - low + 1)
        mu[s] = n
    mu /= mu.sum()

    M -= mu[None, :]

    C = Config()
    C.length_corridors = [S]*1
    C.max_move = A
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
    OLS = np.linalg.pinv(X.T@X)@X.T@y



    U, L, V = np.linalg.svd(M, full_matrices=False)
    U_OLS, S_OLS, V_OLS = np.linalg.svd(OLS, full_matrices=False)

    overlap_l.append(U[:,0].T@U_OLS[:,0])

    hidden = X@U@np.diag(L)

    if plot or S==S_l[-1]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(M)
        ax1.set_title("M")
        ax2.imshow(OLS)
        ax2.set_title("OLS")
        plt.tight_layout()
        plt.show()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.plot(L, marker='o', label='M', c='tab:blue')
        ax1.plot(S_OLS, marker='o', label='OLS', ls='--', c='tab:blue')
        ax1.legend()
        ax1.set_title('singular values')
        ax2.plot(U[:,0], c='tab:blue', label='1st singular vector')
        ax2.plot(U_OLS[:,0], ls='--', c='tab:blue')
        ax2.plot(U[:,1], c='tab:orange', label='2nd singular vector')
        ax2.plot(U_OLS[:,1], ls='--', c='tab:orange')
        ax2.set_title('singular vectors')
        ax2.legend()
        ax3.scatter(hidden[:,0], hidden[:,1], c=loc_y, cmap='coolwarm', marker='o')
        ax3.axis('equal')
        ax3.set_title('hidden states')
        plt.tight_layout()
        plt.show()
overlap_l = abs(np.array(overlap_l))

plt.plot(S_l, 1-overlap_l)
plt.yscale('log')
plt.ylabel('1st singular value overlap')
plt.xlabel('S')
plt.show()

# J = np.concatenate([np.eye(S), np.zeros((S, d1))], axis=1)

# OLS_o = U_OLS[:,[0]]@V_OLS[[0],:]
# plt.imshow(OLS_o)
# plt.show()

# plt.plot(OLS_o)


overlap_l = []
PR_l = []
PR_OLS_l = []
r_2_l = []
r_2_ols_l = []
mse_l = []
mse_ols_l = []
matrix_sim_l = []
S = 100
A_l = np.arange(1, S-1)
for A in tqdm(A_l):
    # S = 100
    A = A
    d1 = 2 * A + 1
    d2 = S
    d = d1 + d2

    M = np.zeros((d, S))
    for s in range(S):
        low = max(0, s-A)
        high = min(S-1, s+A)
        prob = 1 / (high - low + 1)
        M[s, low:high+1] = prob

    for a_i, a in enumerate(range(-A,A+1)):
        low = max(0, a)
        high = min(S-1, S+a-1)
        prob = 1 / (high - low + 1)
        M[d2 + a_i, low:high+1] = prob

    mu = np.zeros(d2)
    for s in range(S):
        low = max(0, s-A)
        high = min(S-1, s+A)
        n = (high - low + 1)
        mu[s] = n
    mu /= mu.sum()

    M -= mu[None, :]

    C = Config()
    C.length_corridors = [S]*1
    C.max_move = A
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
    OLS = np.linalg.pinv(X.T@X + 1e-1*np.eye(X.shape[1]))@X.T@y



    U, L, V = np.linalg.svd(M, full_matrices=False)
    U_OLS, S_OLS, V_OLS = np.linalg.svd(OLS, full_matrices=False)

    overlap_l.append(U[:,0].T@U_OLS[:,0])
    PR_l.append((L**2).sum()**2/(L**4).sum())
    PR_OLS_l.append((S_OLS**2).sum()**2/(S_OLS**4).sum())
    mse_l.append(np.mean((X@M - y)**2))
    mse_ols_l.append(np.mean((X@OLS - y)**2))

    hidden = X@U@np.diag(L)
    r_2_l.append(get_r_2(hidden[:,:2], loc_y))
    hidden_OLS = X@U_OLS@np.diag(S_OLS)
    r_2_ols_l.append(get_r_2(hidden_OLS[:,:2], loc_y))

    matrix_sim_l.append(np.linalg.norm(M-OLS, ord=2)/np.linalg.norm(OLS, ord='fro'))

    if A==1 or A==S//2 or A==S-1:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'A = {A}')
        ax1.plot(L, marker='o', label='M', c='tab:blue')
        ax1.plot(S_OLS, marker='o', label='OLS', ls='--', c='tab:blue')
        ax1.legend()
        ax1.set_title('singular values')
        ax2.plot(U[:,0], c='tab:blue', label='1st singular vector')
        ax2.plot(U_OLS[:,0], ls='--', c='tab:blue')
        ax2.plot(U[:,1], c='tab:orange', label='2nd singular vector')
        ax2.plot(U_OLS[:,1], ls='--', c='tab:orange')
        ax2.set_title('singular vectors')
        ax2.legend()
        ax3.scatter(hidden[:,0], hidden[:,1], c=loc_y, cmap='coolwarm', marker='o')
        ax3.axis('equal')
        ax3.set_title('hidden states')
        plt.tight_layout()
        plt.show()

overlap_l = abs(np.array(overlap_l))
mse_l = np.array(mse_l)
mse_ols_l = np.array(mse_ols_l)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))   
ax1.plot(A_l, 1-overlap_l)
ax1.set_yscale('log')
ax1.set_ylabel('1st singular value overlap')
ax1.set_xlabel('A')
ax2.plot(A_l, PR_l, c='tab:blue')
ax2.plot(A_l, PR_OLS_l, ls='--', c='tab:blue')
ax2.set_ylabel('PR')
ax2.set_xlabel('A')
twinx = ax2.twinx()
twinx.plot(A_l, r_2_l, c='tab:orange')
twinx.plot(A_l, r_2_ols_l, c='tab:orange', ls='--')
twinx.set_ylabel('r^2')
ax3.plot(A_l, mse_l, c='tab:blue')
ax3.plot(A_l, mse_ols_l, c='tab:blue', ls='--')
ax3.set_ylabel('MSE')
ax3.set_yscale('log')
ax3.set_xlabel('A')
ax4.plot(A_l, matrix_sim_l, c='tab:green')
ax4.set_ylabel('matrix distance')
ax4.set_xlabel('A')
plt.tight_layout()
plt.show()


S = 100
A = S//2
C = Config()
C.length_corridors = [S]*1
C.max_move = A
X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
M = np.zeros((X.shape[1], y.shape[1]))
for i in range(X.shape[1]):
    for j in range(y.shape[1]):
        M[i,j] = (X[:,i]*y[:,j]).sum()/X[:,i].sum() - y[:,j].mean()

A_G = X.T@X
A_G[S:,:S] *= -1
A_G[:S, S:] *= -1
plt.imshow(A_G)
plt.show()
OLS_G = np.linalg.pinv(A_G)@X.T@y
OLS = np.linalg.pinv(X.T@X)@X.T@y
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(M)
ax1.set_title('M')
ax2.imshow(OLS_G)
ax2.set_title('OLS_G')
plt.tight_layout()
plt.show()
UG, LG, VG = np.linalg.svd(OLS_G, full_matrices=False)
Um, Lm, Vm = np.linalg.svd(M, full_matrices=False)
U, L, V = np.linalg.svd(OLS, full_matrices=False)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(LG[:20], marker='o')
ax1.plot(Lm[:20], marker='o')
ax1.plot(L[:20], marker='o')
ax1.set_title('L')
ax2.plot(UG[:,0])
ax2.plot(Um[:,0])
ax2.plot(U[:,0])
ax2.set_title('U[:,0]')
plt.tight_layout()
plt.show()

Ux, Lx, Vx = np.linalg.svd(X.T@X, full_matrices=False)
X_null = Ux[:,[-1]]@Vx[[-1],:]

np.linalg.norm(X_null@M)

