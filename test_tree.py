import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


os.makedirs('test_tree', exist_ok=True)

def path_between_indices(a: int, b: int) -> list[int]:
    """
    Return the shortest path from node a to node b in a binary tree
    indexed like:
        1
      2   3
     4 5 6 7

    Actions:
        0 = up
        1 = down left
        2 = down right
    """
    if a < 1 or b < 1:
        raise ValueError("Indices must be positive integers.")

    # Build ancestor paths from each node to root
    path_a = []
    x = a
    while x >= 1:
        path_a.append(x)
        x //= 2

    path_b = []
    x = b
    while x >= 1:
        path_b.append(x)
        x //= 2

    # Reverse so they go root -> node
    path_a.reverse()
    path_b.reverse()

    # Find first index where they differ
    i = 0
    while i < len(path_a) and i < len(path_b) and path_a[i] == path_b[i]:
        i += 1

    # path_a[i-1] is the LCA
    lca_depth = i

    # Steps up from a to LCA
    up_moves = [0] * (len(path_a) - lca_depth)

    # Steps down from LCA to b
    down_moves = []
    for node in path_b[lca_depth:]:
        if node % 2 == 0:
            down_moves.append(1)  # left child
        else:
            down_moves.append(2)  # right child

    return up_moves + down_moves


class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_hidden_output_layers=1, bias=True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.W_ih = nn.ParameterList()
        self.biases = nn.ParameterList() if bias else None
        
        for layer in range(num_layers):
            in_dim = input_size if layer == 0 else hidden_size
            
            self.W_ih.append(
                nn.Parameter(torch.randn(hidden_size, in_dim) * 0.9/np.sqrt(in_dim))
            )
            
            if bias:
                self.biases.append(
                    nn.Parameter(torch.zeros(hidden_size))
                )
        
        hidden_output_layers = nn.ModuleList()
        for _ in range(num_hidden_output_layers):
            hidden_output_layers.append(nn.Linear(hidden_size, hidden_size, bias=bias))
            hidden_output_layers.append(nn.ReLU())
        self.hidden_output_layers = nn.Sequential(*hidden_output_layers)
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size, bias=bias))

    def forward(self, x, h0=None):
        """
        batch_first=True

        x:  (batch_size, seq_len, input_size)
        h0: (num_layers, batch_size, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # IMPORTANT: avoid in-place writes into a tensor view (e.g. h[layer] = ...)
        # because it breaks autograd versioning. Keep h as a Python list and stack
        # only at the end.
        if h0 is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h = [h0[layer] for layer in range(self.num_layers)]

        outputs = []
        hidden_states = []
        for t in range(seq_len):
            input_t = x[:, t, :]  # (B, in_dim)

            for layer in range(self.num_layers):
                prev_h = h[layer]  # (B, H)

                linear = (
                    input_t @ self.W_ih[layer].T
                    + prev_h
                )

                if self.biases is not None:
                    linear = linear + self.biases[layer]  # broadcast over batch

                h[layer] = F.relu(linear) if True else linear
                input_t = h[layer]  # input to next layer
            h[-1] = self.hidden_output_layers(h[-1])
            hidden_states.append(h[-1])
            outputs.append(self.output_layer(h[-1]))
        
        hidden_states = torch.stack(hidden_states, dim=1)  # (B, T, H)
        outputs = torch.stack(outputs, dim=1)  # (B, T, H)
        
        return outputs, hidden_states

def depth(i: int) -> int:
    """Compute depth of node i (root has depth 0)."""
    d = 0
    while i > 0:
        i = (i - 1) // 2
        d += 1
    return d


def lca(i: int, j: int) -> int:
    """Compute lowest common ancestor using parent climbing."""
    di, dj = depth(i), depth(j)

    # Bring both nodes to same depth
    while di > dj:
        i = (i - 1) // 2
        di -= 1
    while dj > di:
        j = (j - 1) // 2
        dj -= 1

    # Climb together until equal
    while i != j:
        i = (i - 1) // 2
        j = (j - 1) // 2

    return i


def tree_distance(i: int, j: int) -> int:
    """Distance between nodes i and j."""
    i-=1
    j-=1
    di = depth(i)
    dj = depth(j)
    ancestor = lca(i, j)
    da = depth(ancestor)

    return di + dj - 2 * da

class Tree():
    def __init__(self, d, k):
        self.d = d
        self.states = np.arange(1, 2**d)
        self.n_states = len(self.states)
        self.actions = [0, 1, 2, 3]
        self.T_depth = int(np.log2(self.n_states)) + 1
        self.states_in = np.eye(self.n_states)
        self.actions_in = np.eye(len(self.actions))
        self.k = k
        self.distance_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.distance_matrix[i, j] = tree_distance(i+1, j+1)
    
    def walk(self, state_start, i=None):
        if i is None:
            target_state = np.random.choice(np.where(self.distance_matrix[state_start-1] <= self.k)[0])+1
        else:
            nghbr_states = np.where(self.distance_matrix[state_start-1] <= self.k)[0]
            target_state = nghbr_states[i%len(nghbr_states)]+1
        path = path_between_indices(state_start, target_state)
        X_seq = []
        y_seq = []
        loc_X_seq = []
        loc_y_seq = []
        action_taken_seq = []
        state_curr = state_start
        for i in range(self.k):
            state_next, action = self.take_action(state_curr, path[i] if i < len(path) else None)
            X_seq.append(np.concatenate([(i==0)*self.states_in[state_curr-1], self.actions_in[action]]))
            y_seq.append(self.states_in[state_curr-1])
            loc_X_seq.append(state_curr)
            loc_y_seq.append(state_curr)
            action_taken_seq.append(action)
            state_curr = state_next
        return X_seq, y_seq, loc_X_seq, loc_y_seq, action_taken_seq
        
    def take_action(self, state_curr, action=None):
        if action is not None:
            next_state = self.move(state_curr, action)
        else:
            next_state = -1
            while next_state not in self.states:
                action = np.random.choice(self.actions)
                next_state = self.move(state_curr, action)
        if next_state not in self.states:
            raise ValueError(f'Next state {next_state} not in states, start state: {state_curr}, action: {action}')
        return next_state, action
    
    def move(self, state, action):
        if action == 0:
            next_state = state//2
        elif action == 1:
            next_state = state*2
        elif action == 2:
            next_state = state*2 + 1
        elif action == 3:
            next_state = state
        return next_state

d = 3
sim_l = []
sim_mat_l = []
matrices_l = []
loss_l_l = []
k_l = np.arange(1, 2*(d-1)+1)
for k in k_l:
    print(f'############ k: {k} / {k_l[-1]} ############')
    # A = 4
    tree = Tree(d, k)
    X = []
    y = []
    loc_X = []
    loc_y = []
    action_taken = []
    for state in tree.states:
        for i in range(2**d-1):
            X_seq, y_seq, loc_X_seq, loc_y_seq, action_taken_seq = tree.walk(state, i)
            X.append(X_seq)
            y.append(y_seq)
            loc_X.append(loc_X_seq)
            loc_y.append(loc_y_seq)
            action_taken.append(action_taken_seq)

    X = np.array(X)
    y = np.array(y)
    loc_X = np.array(loc_X)
    loc_y = np.array(loc_y)
    action_taken = np.array(action_taken)
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')

    # Example data sample
    i = np.random.choice(len(X))
    print(f'X[{i}]: {X[i]}')
    print(f'y[{i}]: {y[i]}')
    print(f'loc_X[{i}]: {loc_X[i]}')
    print(f'loc_y[{i}]: {loc_y[i]}')
    print(f'action_taken[{i}]: {action_taken[i]}')

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Prepare data for torch
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    # Create model
    input_size = X.shape[2]
    output_size = y.shape[2]
    hidden_size = 512
    n_layers = 5
    model = LinearRNN(input_size, hidden_size, output_size, n_layers).to(device)
    initial_weights = deepcopy(model.state_dict())
    with torch.no_grad():
        outputs, hidden_states = model(X_tensor)
        print(f'Sig_2 of last hidden: {hidden_states[-1].var().item()}')

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    y_var = y_tensor.var().cpu()
    # Training loop
    loss_l = []
    accuracy_l = []
    for epoch in tqdm(range(10000)):
        optimizer.zero_grad()
        outputs, hidden = model(X_tensor)
        if isinstance(criterion, nn.MSELoss):
            loss = criterion(outputs, y_tensor)
        else:
            loss = criterion(outputs.view(-1, outputs.size(-1)), y_tensor.argmax(-1).view(-1))
        loss.backward()
        optimizer.step()
        loss_l.append(loss.item()/y_var)
        accuracy_l.append((outputs.argmax(dim=-1) == y_tensor.argmax(dim=-1)).float().mean().item())

    # Testing
    with torch.no_grad():
        outputs, hidden_states = model(X_tensor)
    # print(criterion(outputs, y).item()/y_var)

    only_first_step = True
    if only_first_step:
        X = X_tensor.cpu().numpy()[:,0,:]
        y = y_tensor.cpu().numpy()[:,0,:]
        h_np = hidden_states.detach().cpu().numpy()[:,0,:]
        loc_y = loc_y[:,0]
        loc_X = loc_X[:,0]
        action_taken = action_taken[:,0]
    else:
        X = X_tensor.cpu().numpy().reshape(-1, X_tensor.shape[2])  # Convert to numpy array if X is a torch tensor
        y = y_tensor.cpu().numpy().reshape(-1, y_tensor.shape[2])  # Convert to numpy array if y is a torch tensor
        h_np = hidden_states.detach().cpu().numpy().reshape(-1, hidden_states.shape[-1])  # Convert to numpy array if hidden is a torch tensor
        loc_y = loc_y.reshape(-1)
        loc_X = loc_X.reshape(-1)
        action_taken = action_taken.reshape(-1)


    # plt.plot(loss_l)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.title("Loss over time")
    # plt.show()
    # PCA on last layer activations
    # pca = PCA(n_components=2)
    # act_pca = pca.fit_transform(activations_np)

    # plt.figure(figsize=(8,6))
    # plt.scatter(act_pca[:,0], act_pca[:,1], c=depth_X, cmap='viridis', s=10, alpha=0.8)
    # plt.colorbar(label='Tree Depth')
    # plt.title("PCA of Last Layer Activations")
    # plt.xlabel('PC 1')
    # plt.ylabel('PC 2')
    # plt.tight_layout()
    # plt.show()

    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    import seaborn as sns

    # Calculate the pairwise distance matrix of last hidden layer activations
    distance_matrix = squareform(pdist(h_np, metric='euclidean'))

    distance_matrix_states = np.zeros_like(distance_matrix)
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            distance_matrix_states[i, j] = tree_distance(loc_y[i], loc_y[j])

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix_states, method='ward')

    # Sort the distance matrix by the clustering to reveal clusters

    distance_matrix_inputs = squareform(pdist(X, metric='euclidean'))
    distance_matrix_outputs = squareform(pdist(y, metric='euclidean'))

    order = leaves_list(linkage_matrix)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    (ax1, ax2, ax3, ax4) = axs.flatten()
    ax1.imshow(distance_matrix[order][:, order], cmap='viridis')
    ax1.set_title("hidden")
    ax2.imshow(distance_matrix_states[order][:, order], cmap='viridis')
    ax2.set_title("tree")
    ax3.imshow(distance_matrix_inputs[order][:, order], cmap='viridis')
    ax3.set_title("inputs")
    ax4.imshow(distance_matrix_outputs[order][:, order], cmap='viridis')
    ax4.set_title("output")
    plt.show()

    # Compute the similarity (correlation) between all 4 distance matrices and plot using imshow

    from scipy.stats import pearsonr


    # The order used for all matrices:
    dm_hidden = distance_matrix[order][:, order]
    dm_tree = distance_matrix_states[order][:, order]
    # dm_inputs = distance_matrix_inputs[order][:, order]
    # dm_outputs = distance_matrix_outputs[order][:, order]
    
    # Filter all distance matrices
    filter = action_taken[order] <= 1
    dm_hidden = dm_hidden[filter][:, filter]
    dm_tree = dm_tree[filter][:, filter]
    # dm_inputs = dm_inputs[filter][:, filter]
    # dm_outputs = dm_outputs[filter][:, filter]

    # Since distance matrices are symmetric with zeros on diagonal, ignore diagonal and duplicate entries
    def extract_upper_triangular_values(mat):
        return mat[np.triu_indices_from(mat, k=1)]

    mat_names = ['Hidden', 'Tree', 'Inputs', 'Outputs']
    matrices = [
        dm_hidden,
        dm_tree,
        # dm_inputs,
        # dm_outputs
    ]
    vecs = [extract_upper_triangular_values(m) for m in matrices]

    # Create a similarity (correlation) matrix
    num = len(vecs)
    similarity = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if i == j:
                similarity[i, j] = 1.0
            else:
                # Compute Pearson correlation between the upper-triangle values
                similarity[i, j], _ = pearsonr(vecs[i], vecs[j])

    # plt.figure(figsize=(6,5))
    # im = plt.imshow(similarity, cmap="coolwarm", vmin=-1, vmax=1)
    # plt.colorbar(im, label="Pearson correlation")
    # plt.xticks(range(num), mat_names, rotation=45)
    # plt.yticks(range(num), mat_names)
    # plt.title("Similarity between distance matrices")
    # plt.tight_layout()
    # plt.show()
    print(f'Loss: {loss.item()/y_var}, Accuracy: {accuracy_l[-1]}')

    sim_l.append(similarity[0,1])
    sim_mat_l.append(similarity)
    matrices_l.append(matrices)
    loss_l_l.append(loss_l)

fig = plt.figure(figsize=(10, 5))
plt.plot(sim_l, marker='o')
plt.ylim(0, 1)
fig.savefig('test_tree/sim_l.png')
plt.show()

# inputs_states_sim = [sim_mat[2,1] for sim_mat in sim_mat_l]
# inputs_hidden_sim = [sim_mat[0,2] for sim_mat in sim_mat_l]
hidden_states_sim = [sim_mat[0,1] for sim_mat in sim_mat_l]

fig = plt.figure(figsize=(10, 5))
# plt.plot(inputs_states_sim, marker='o', label='Inputs vs States')
# plt.plot(inputs_hidden_sim, marker='o', label='Inputs vs Hidden')
plt.plot(hidden_states_sim, marker='o', label='Hidden vs States')
plt.legend()
fig.savefig('test_tree/inputs_states_hidden_sim.png')
plt.show()

state_distance = distance_matrix_states[np.triu_indices_from(distance_matrix_states, k=1)]
hidden_distance = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
import pandas as pd

# Make sure state_distance and hidden_distance are numpy arrays of same length
# Group hidden_distance values according to each unique value in state_distance
df = pd.DataFrame({'state_distance': state_distance, 'hidden_distance': hidden_distance})

# Sort by state_distance for nicer plots (optional)
df = df.sort_values('state_distance')

unique_states = np.unique(state_distance)
data_to_plot = [df[df['state_distance'] == val]['hidden_distance'].values for val in unique_states]

fig = plt.figure(figsize=(10, 5))
plt.boxplot(data_to_plot, positions=unique_states)
plt.xlabel('State Distance')
plt.ylabel('Hidden Distance')
plt.title('Hidden Distance Distribution for Each State Distance')
plt.xticks(unique_states.astype(int), unique_states.astype(int))  
fig.savefig('test_tree/hidden_distance_distribution_boxplot.png')
plt.show()

fig = plt.figure(figsize=(10, 5))
for i, matrices in enumerate(matrices_l):
    hidden_distance = matrices[0][np.triu_indices_from(matrices[0], k=1)]
    state_distance = matrices[1][np.triu_indices_from(matrices[1], k=1)]
    # Make sure state_distance and hidden_distance are numpy arrays of same length
    # Group hidden_distance values according to each unique value in state_distance
    df = pd.DataFrame({'state_distance': state_distance, 'hidden_distance': hidden_distance})

    # Sort by state_distance for nicer plots (optional)
    df = df.sort_values('state_distance')

    unique_states = np.unique(state_distance)
    data_to_plot = [df[df['state_distance'] == val]['hidden_distance'].values for val in unique_states]
    plt.plot([d.mean() for d in data_to_plot], marker='o', label=f'A={i+1}')
plt.legend()
fig.savefig('test_tree/hidden_distance_distribution.png')
plt.show()

fig = plt.figure(figsize=(10, 5))
for i, loss_l in enumerate(loss_l_l):
    plt.plot(loss_l, label=f'A={i+1}')
plt.axhline(1, ls='--', c='k', alpha=0.5)
plt.yscale('log')
plt.legend()
fig.savefig('test_tree/loss.png')
plt.show()

from scipy.stats import spearmanr

corr_l = []
coverage_l = []
for i, matrices in enumerate(matrices_l):
    hidden_distance = matrices[0][np.triu_indices_from(matrices[0], k=1)]
    state_distance = matrices[1][np.triu_indices_from(matrices[1], k=1)]
    coverage_l.append((state_distance <= k_l[i]).mean())
    corr_l.append(spearmanr(hidden_distance, state_distance).correlation)
fig = plt.figure(figsize=(10, 5))
plt.plot(coverage_l, corr_l, marker='o')
plt.ylim(-0.1, 1)
plt.xlim(-0,1.1)
fig.savefig('test_tree/coverage_corr.png')
plt.show()