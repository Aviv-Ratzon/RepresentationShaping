import numpy as np
from tqdm import tqdm

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

def Tree():
    def __init__(self, d, k):
        self.d = d
        self.states = np.arange(1, 2**d)
        self.n_states = len(self.states)
        self.actions = [0, 1, 2]
        self.T_depth = int(np.log2(self.n_states)) + 1
        self.states_in = np.eye(self.n_states)
        self.actions_in = np.eye(len(self.actions))
        self.k = k
    
    def walk(self, state_start):
        X_seq = []
        y_seq = []
        loc_X_seq = []
        loc_y_seq = []
        depth_X_seq = []
        depth_y_seq = []
        action_taken_seq = []
        direction_taken_seq = []
        state_curr = state_start
        for _ in range(self.k):
            state_next, action = self.take_action(state_curr)
            X_seq.append(np.concatenate([self.states_in[state_curr], self.actions_in[action]]))
            y_seq.append(self.states_in[state_next])
            loc_X_seq.append(state_curr)
            loc_y_seq.append(state_next)
            depth_X_seq.append(int(np.log2(state_curr)) + 1)
            depth_y_seq.append(int(np.log2(state_next)) + 1)
            action_taken_seq.append(action)
            direction_taken_seq.append(direction)
            state_curr = state_next
        return X_seq, y_seq, loc_X_seq, loc_y_seq, depth_X_seq, depth_y_seq, action_taken_seq, direction_taken_seq
        
    def take_action(self, state_curr):
        next_state = -1
        while next_state not in self.states:
            action = np.random.choice(self.actions)
            if action == 0:
                state_next = state_curr//2
            elif action == 1:
                state_next = state_curr*2
            elif action == 2:
                state_next = state_curr*2 + 1
        return state_next, action
    


d = 4
sim_l = []
sim_mat_l = []
matrices_l = []
loss_l_l = []
for A in range(1, d):
    # A = 4
    states = np.arange(1, 2**d)
    n_states = len(states)
    actions = np.arange(0, A+1)
    T_depth = int(np.log2(n_states)) + 1
    states_in = np.eye(n_states)
    actions_in = np.eye(A*3+1)

    def get_actions_in(action, direction_idx):
        idx = 0 if action == 0 else direction_idx*A + action
        return actions_in[idx]

    X = []
    y = []
    loc_X = []
    loc_y = []
    depth_X = []
    depth_y = []
    action_taken = []
    direction_taken = []
    for state in states:
        s_depth = int(np.log2(state)) + 1
        for action in actions:
            for direction_idx, direction in enumerate(['up', 'left', 'right']):
                direction_val = -1 if direction == 'up' else 1
                if s_depth + action*direction_val < 1 or s_depth + action*direction_val > T_depth or (action == 0 and direction_idx != 0):
                    continue
                s_next = state
                for _ in range(action):
                    if direction == 'up':
                        s_next = s_next//2
                    elif direction == 'left':
                        s_next = s_next*2
                    elif direction == 'right':
                        s_next = s_next*2 + 1
                        
                X.append(np.concatenate([states_in[state-1], get_actions_in(action, direction_idx)]))
                y.append(states_in[s_next-1])
                loc_X.append(state)
                loc_y.append(s_next)
                depth_X.append(s_depth)
                depth_y.append(int(np.log2(s_next)) + 1)
                action_taken.append(action)
                direction_taken.append(direction_idx)
    X = np.array(X)
    y = np.array(y)
    loc_X = np.array(loc_X)
    loc_y = np.array(loc_y)
    depth_X = np.array(depth_X)
    depth_y = np.array(depth_y)
    action_taken = np.array(action_taken)
    direction_taken = np.array(direction_taken)
    print(f'X shape: {X.shape}')
    print(f'y shape: {y.shape}')

    # Example data sample
    i = np.random.choice(len(X))
    print(f'X[{i}]: {X[i]}')
    print(f'y[{i}]: {y[i]}')
    print(f'loc_X[{i}]: {loc_X[i]}')
    print(f'loc_y[{i}]: {loc_y[i]}')
    print(f'depth_X[{i}]: {depth_X[i]}')
    print(f'depth_y[{i}]: {depth_y[i]}')
    print(f'action_taken[{i}]: {action_taken[i]}')
    print(f'direction_taken[{i}]: {direction_taken[i]}')

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Prepare data for torch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    class FeedForwardNN(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim, init_scale=0.01):
            super(FeedForwardNN, self).__init__()
            layers = []
            prev_dim = input_dim
            self.init_scale = init_scale
            for h in hidden_dims:
                linear = nn.Linear(prev_dim, h)
                nn.init.normal_(linear.weight, mean=0.0, std=self.init_scale)
                nn.init.constant_(linear.bias, 0.0)
                layers.append(linear)
                layers.append(nn.ReLU())
                prev_dim = h
            self.hidden = nn.Sequential(*layers)
            self.out = nn.Linear(prev_dim, output_dim)
            # Do not apply init_scale to output layer -- use default init
        def forward(self, x):
            z = self.hidden(x)
            out = self.out(z)
            return out, z

    input_dim = X.shape[1]
    hidden_dims = [1024]*5
    output_dim = y.shape[1]
    model = FeedForwardNN(input_dim, hidden_dims, output_dim)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)

    # Training loop
    loss_l = []
    epochs = 10000
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        output, _ = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            # print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            loss_l.append(loss.item()/y.var().item())
    # Get last hidden layer activations
    with torch.no_grad():
        _, activations = model(X_tensor)
        activations_np = activations.detach().numpy()

    accuracy = (output.argmax(dim=-1) == y_tensor.argmax(dim=-1)).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}, loss: {loss.item()/y.var().item():.4f}")

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
    distance_matrix = squareform(pdist(activations_np, metric='euclidean'))

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
    dm_inputs = distance_matrix_inputs[order][:, order]
    dm_outputs = distance_matrix_outputs[order][:, order]
    
    # Filter all distance matrices
    filter = action_taken[order] <= 1
    dm_hidden = dm_hidden[filter][:, filter]
    dm_tree = dm_tree[filter][:, filter]
    dm_inputs = dm_inputs[filter][:, filter]
    dm_outputs = dm_outputs[filter][:, filter]

    # Since distance matrices are symmetric with zeros on diagonal, ignore diagonal and duplicate entries
    def extract_upper_triangular_values(mat):
        return mat[np.triu_indices_from(mat, k=1)]

    mat_names = ['Hidden', 'Tree', 'Inputs', 'Outputs']
    matrices = [
        dm_hidden,
        dm_tree,
        dm_inputs,
        dm_outputs
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

    sim_l.append(similarity[0,1])
    sim_mat_l.append(similarity)
    matrices_l.append(matrices)
    loss_l_l.append(loss_l)

plt.plot(sim_l, marker='o')
plt.ylim(0, 1)

inputs_states_sim = [sim_mat[2,1] for sim_mat in sim_mat_l]
inputs_hidden_sim = [sim_mat[0,2] for sim_mat in sim_mat_l]
hidden_states_sim = [sim_mat[0,1] for sim_mat in sim_mat_l]

plt.plot(inputs_states_sim, marker='o', label='Inputs vs States')
plt.plot(inputs_hidden_sim, marker='o', label='Inputs vs Hidden')
plt.plot(hidden_states_sim, marker='o', label='Hidden vs States')
plt.legend()
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

plt.boxplot(data_to_plot, positions=unique_states)
plt.xlabel('State Distance')
plt.ylabel('Hidden Distance')
plt.title('Hidden Distance Distribution for Each State Distance')
plt.xticks(unique_states.astype(int), unique_states.astype(int))  
plt.show()

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
plt.show()

for i, loss_l in enumerate(loss_l_l):
    plt.plot(loss_l, label=f'A={i+1}')
plt.yscale('log')
plt.legend()
plt.show()