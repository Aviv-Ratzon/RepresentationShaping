from utils import calc_PR, get_r_2
import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _shared_prefix_length(a: int, b: int, depth: int) -> int:
    """
    Shared prefix length of two leaf IDs in a balanced binary tree.
    Leaves are assumed indexed from 0 to 2^depth - 1.
    """
    x = a ^ b
    if x == 0:
        return depth
    return depth - x.bit_length()


def _tree_distance_matrix_from_leaf_ids(leaf_ids: np.ndarray, depth: int) -> np.ndarray:
    """
    Exact tree distance between leaves in a balanced binary tree.

    Parameters
    ----------
    leaf_ids : array of shape (n_samples,)
        Leaf indices in [0, 2^depth - 1].
    depth : int
        Depth of the balanced binary tree. Root depth = 0; leaves at 'depth'.

    Returns
    -------
    D_tree : array of shape (n_samples, n_samples)
        Tree distances in number of edges.
    """
    leaf_ids = np.asarray(leaf_ids, dtype=np.int64)
    n = len(leaf_ids)
    D_tree = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            lcp = _shared_prefix_length(int(leaf_ids[i]), int(leaf_ids[j]), depth)
            dist = 2 * (depth - lcp)
            D_tree[i, j] = dist
            D_tree[j, i] = dist
    return D_tree


def _tree_distance_heap_index(u: int, v: int) -> int:
    """
    Tree distance between two nodes in a binary tree indexed in heap order:
    root=0, children of i are 2*i+1 and 2*i+2.
    """
    dist = 0
    uu, vv = u, v
    while uu != vv:
        if uu > vv:
            uu = (uu - 1) // 2
        else:
            vv = (vv - 1) // 2
        dist += 1
    return dist


def _tree_distance_matrix_from_heap_indices(node_indices: np.ndarray) -> np.ndarray:
    """
    Exact tree distance between arbitrary nodes using heap indexing.
    """
    node_indices = np.asarray(node_indices, dtype=np.int64)
    n = len(node_indices)
    D_tree = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            dist = _tree_distance_heap_index(int(node_indices[i]), int(node_indices[j]))
            D_tree[i, j] = dist
            D_tree[j, i] = dist
    return D_tree


def _build_mutual_knn_geodesic_distances(
    X: np.ndarray,
    k: int = 10,
    pca_components: int | None = None,
    pca_var: float | None = 0.95,
    scale_features: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Build a mutual k-NN graph and compute all-pairs shortest-path distances.

    Notes
    -----
    - Local Euclidean distances are only used to stitch together the graph locally.
    - The final metric is graph geodesic distance, not ambient Euclidean distance.
    - If the graph is disconnected, k is increased until connected.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    if scale_features:
        X_proc = StandardScaler().fit_transform(X)
    else:
        X_proc = X.copy()

    # Mild denoising can help in the small-n, moderate-d regime.
    if pca_components is not None:
        X_proc = PCA(n_components=pca_components, svd_solver="full").fit_transform(X_proc)
    elif pca_var is not None and d > 8:
        X_proc = PCA(n_components=pca_var, svd_solver="full").fit_transform(X_proc)

    k = max(3, min(k, n - 1))

    while True:
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nbrs.fit(X_proc)
        distances, indices = nbrs.kneighbors(X_proc)

        # Remove self-neighbor at column 0
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Local scaling for robustness to density variation
        sigma = distances[:, -1] + 1e-12

        rows, cols, vals = [], [], []
        neighbor_sets = [set(row) for row in indices]

        for i in range(n):
            for pos, j in enumerate(indices[i]):
                j = int(j)
                if i in neighbor_sets[j]:  # mutual k-NN
                    dij = distances[i, pos]
                    # Edge length for graph shortest paths.
                    # Normalized local metric improves stability across densities.
                    length = dij / np.sqrt(sigma[i] * sigma[j])

                    rows.append(i)
                    cols.append(j)
                    vals.append(length)
                    rows.append(j)
                    cols.append(i)
                    vals.append(length)

        G = csr_matrix((vals, (rows, cols)), shape=(n, n))
        n_components, _ = connected_components(G, directed=False)

        if n_components == 1:
            D_geo = shortest_path(G, directed=False, unweighted=False)
            return D_geo, k

        if k >= n - 1:
            raise RuntimeError("Could not build a connected graph even with k = n - 1.")
        k = min(k + 2, n - 1)


def tree_manifold_score(
    hidden_reps: np.ndarray,
    node_indices: np.ndarray,
    *,
    index_type: str = "leaf",
    tree_depth: int | None = None,
    initial_k: int = 10,
    pca_components: int | None = None,
    pca_var: float | None = 0.95,
    n_permutations: int = 0,
    random_state: int = 0,
) -> dict:
    """
    Measure how well hidden representations preserve balanced-binary-tree structure.

    Parameters
    ----------
    hidden_reps : array, shape (n_samples, n_hidden_dims)
        Hidden representations for each sample.
    node_indices : array, shape (n_samples,)
        Tree labels for each sample.

        If index_type='leaf':
            node_indices are leaf IDs in [0, 2^tree_depth - 1].

        If index_type='heap':
            node_indices use heap indexing over the full binary tree:
            root=0, children=2*i+1, 2*i+2.
    index_type : {'leaf', 'heap'}
        How the tree node indices are encoded.
    tree_depth : int, optional
        Required if index_type='leaf'.
    initial_k : int
        Initial k for mutual k-NN graph. It will be increased automatically if needed.
    pca_components : int or None
        Explicit PCA dimension before graph construction.
    pca_var : float or None
        If pca_components is None, keep this explained variance fraction with PCA.
    n_permutations : int
        Optional permutation test. 0 disables it.
    random_state : int
        RNG seed for permutation test.

    Returns
    -------
    result : dict
        Contains:
        - 'score_spearman': main Tree Manifold Score
        - 'p_value_perm': permutation p-value or None
        - 'k_used': final k used in graph construction
        - 'D_geo': geodesic distance matrix
        - 'D_tree': tree distance matrix
    """
    X = np.asarray(hidden_reps, dtype=np.float64)
    y = np.asarray(node_indices)

    if X.ndim != 2:
        raise ValueError("hidden_reps must have shape (n_samples, n_dims).")
    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("node_indices must have shape (n_samples,).")

    if index_type == "leaf":
        if tree_depth is None:
            raise ValueError("tree_depth is required when index_type='leaf'.")
        D_tree = _tree_distance_matrix_from_leaf_ids(y, tree_depth)
    elif index_type == "heap":
        D_tree = _tree_distance_matrix_from_heap_indices(y)
    else:
        raise ValueError("index_type must be either 'leaf' or 'heap'.")

    D_geo, k_used = _build_mutual_knn_geodesic_distances(
        X,
        k=initial_k,
        pca_components=pca_components,
        pca_var=pca_var,
        scale_features=True,
    )

    triu = np.triu_indices(X.shape[0], k=1)
    geo_vec = D_geo[triu]
    tree_vec = D_tree[triu]

    # Rank correlation is more robust than Pearson here.
    score, _ = spearmanr(geo_vec, tree_vec)

    p_value_perm = None
    if n_permutations > 0:
        rng = np.random.default_rng(random_state)
        null_scores = np.empty(n_permutations, dtype=np.float64)

        for b in range(n_permutations):
            perm = rng.permutation(len(y))
            D_tree_perm = D_tree[perm][:, perm]
            tree_perm_vec = D_tree_perm[triu]
            null_scores[b], _ = spearmanr(geo_vec, tree_perm_vec)

        # One-sided test: higher correlation means better tree preservation
        p_value_perm = (1 + np.sum(null_scores >= score)) / (n_permutations + 1)

    return {
        "score_spearman": float(score),
        "p_value_perm": None if p_value_perm is None else float(p_value_perm),
        "k_used": int(k_used),
        "D_geo": D_geo,
        "D_tree": D_tree,
    }

UP_ACTION = 0
DOWN_LEFT_ACTION = 1
DOWN_RIGHT_ACTION = 2
STAY_ACTION = 3

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda:4'
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


class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_hidden_output_layers=3, init_scale=1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.hidden_layers = []
        for i in range(num_layers):
            in_dim = input_size if i == 0 else hidden_size
            self.hidden_layers.append(nn.Linear(in_dim, hidden_size))
            self.hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*self.hidden_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # initialize hidden layers
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=init_scale)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        hidden = self.hidden_layers(x)
        output = self.output_layer(hidden)
        return output, hidden

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
        self.actions = [0, 1, 2]
        self.T_depth = int(np.log2(self.n_states)) + 1
        self.states_in = np.eye(self.n_states)
        self.actions_in = np.eye(len(self.actions))
        self.k = k
        self.distance_matrix = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.distance_matrix[i, j] = tree_distance(i+1, j+1)
    
    def walk(self):
        n_features = len(self.states) + len(self.actions)*self.k
        X = []
        y = []
        loc_X = []
        loc_y = []
        action_taken = []
        for state_start in tree.states:
            for target_state in self.states:
                path = path_between_indices(state_start, target_state)
                if len(path) > self.k:
                    continue
                state_curr = state_start
                actions_in = []
                actions = []
                for i in range(len(path)):
                    state_curr, action = self.take_action(state_curr, path[i] if i < len(path) else None)
                    actions_in.append(self.actions_in[action])
                    actions.append(action)
                X_seq = self.states_in[state_start-1]
                if len(actions_in) > 0:
                    X_seq = np.concatenate([X_seq, np.concatenate(actions_in)])
                X_seq = np.pad(X_seq, (0, n_features - len(X_seq)), 'constant', constant_values=0)
                y_seq = self.states_in[state_curr-1]
                loc_X_seq = state_start
                loc_y_seq = state_curr
                action_taken_seq = np.pad(actions, (0, self.k - len(actions)), 'constant', constant_values=STAY_ACTION)
                X.append(X_seq)
                y.append(y_seq)
                loc_X.append(loc_X_seq)
                loc_y.append(loc_y_seq)
                action_taken.append(action_taken_seq)
        X = np.stack(X)
        y = np.stack(y)
        loc_X = np.stack(loc_X)
        loc_y = np.stack(loc_y)
        action_taken = np.stack(action_taken)
        return X, y, loc_X, loc_y, action_taken
        
    def take_action(self, state_curr, action=None):
        if action is not None:
            next_state = self.move(state_curr, action)
        else:
            next_state = state_curr
            action = STAY_ACTION
            # next_state = -1
            # while next_state not in self.states:
            #     action = np.random.choice(self.actions)
            #     next_state = self.move(state_curr, action)
        if next_state not in self.states:
            raise ValueError(f'Next state {next_state} not in states, start state: {state_curr}, action: {action}')
        return next_state, action
    
    def move(self, state, action):
        if action == UP_ACTION:
            next_state = state//2
        elif action == DOWN_LEFT_ACTION:
            next_state = state*2
        elif action == DOWN_RIGHT_ACTION:
            next_state = state*2 + 1
        elif action == STAY_ACTION:
            next_state = state
        return next_state

var_name = 'n_layers'
var_values = [5]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
manifold_score_l = []
for var_value in var_values:
    d = 6
    sim_l = []
    sim_mat_l = []
    matrices_l = []
    loss_l_l = []
    accuracy_l_l = []
    PR_l = []
    k_l = np.arange(1, 2*(d-1)+1)
    r2_depth_l = []
    r2_branch_l = []
    r2_subtrees_l = []
    test_accuracy_l = []
    test_loss_l = []
    for k in k_l:
        print(f'#'*20 + f' k: {k} / {k_l[-1]} ' + '#'*20)
        # A = 4
        tree = Tree(d, k)
        X, y, loc_X, loc_y, action_taken = tree.walk()
        omit_loc = loc_y.max()
        omit_inds = np.where((loc_y == omit_loc) | (loc_X==omit_loc))[0]
        omit_inds = np.random.choice(omit_inds, size=len(omit_inds)//2, replace=False)

        X_test = X[omit_inds]
        y_test = y[omit_inds]
        
        X = np.delete(X, omit_inds, axis=0)
        y = np.delete(y, omit_inds, axis=0)
        loc_X = np.delete(loc_X, omit_inds, axis=0)
        loc_y = np.delete(loc_y, omit_inds, axis=0)
        action_taken = np.delete(action_taken, omit_inds, axis=0)

        print(f'X shape: {X.shape} --- test shape: {X_test.shape}')
        print(f'y shape: {y.shape} --- test shape: {y_test.shape}')

        # Example data sample
        i = np.random.choice(len(X))
        print(f'X[{i}]: {X[i]}')
        print(f'y[{i}]: {y[i]}')
        print(f'loc_X[{i}]: {loc_X[i]}, X pos: {X[i][:y.shape[0]].argmax()+1}')
        print(f'loc_y[{i}]: {loc_y[i]}, y pos: {y[i].argmax()+1}')
        print(f'action_taken[{i}]: {action_taken[i]}')

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # Prepare data for torch
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

        X_tensor_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_tensor_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        # Create model
        input_size = X.shape[1]
        output_size = y.shape[1]
        hidden_size = 512
        n_layers = var_value
        model = DeepNet(input_size, hidden_size, output_size, n_layers, init_scale=0.9).to(device)
        initial_weights = deepcopy(model.state_dict())
        with torch.no_grad():
            outputs, hidden_states = model(X_tensor)
            print(f'Sig_2 of last hidden: {hidden_states[-1].var().item()}')

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

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
            loss_l.append(loss.item())
            accuracy_l.append((outputs.argmax(dim=-1) == y_tensor.argmax(dim=-1)).float().mean().item())

        # Testing
        with torch.no_grad():
            outputs, hidden_states = model(X_tensor)
            outputs_test, hidden_states_test = model(X_tensor_test)
            test_loss = criterion(outputs_test, y_tensor_test).item()
            test_accuracy = (outputs_test.argmax(dim=-1) == y_tensor_test.argmax(dim=-1)).float().mean().item()
        # print(criterion(outputs, y).item()/y_var)
        print(f'Loss: {loss.item()}, Accuracy: {accuracy_l[-1]}')
        print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')


        only_stay_actions = np.all(action_taken==STAY_ACTION, axis=1)
        filter = np.ones(len(X), dtype=bool)
        X = X_tensor.cpu().numpy()[filter]
        y = y_tensor.cpu().numpy()[filter]
        h_np = hidden_states.detach().cpu().numpy()[filter]
        loc_y = loc_y[filter]
        loc_X = loc_X[filter]

        loc_depth = np.array([depth(loc_y[i]-1) for i in range(len(loc_y))])
        loc_branch = np.array([loc_y[i]>=(2**loc_depth[i]+2.**(loc_depth[i]-1)) for i in range(len(loc_y))]).astype(int)
        loc_branch[loc_y==1] = 2
        loc_y_subtrees = loc_y.copy()
        loc_y_subtrees[loc_branch==1] = loc_y[loc_branch==1] - 2**(loc_depth[loc_branch==1]-1)

        r2_depth = get_r_2(PCA(n_components=1).fit_transform(h_np), loc_depth)
        r2_branch = get_r_2(PCA(n_components=1).fit_transform(h_np), loc_branch)
        r2_subtrees = get_r_2(PCA(n_components=1).fit_transform(h_np), loc_y_subtrees)


        result = tree_manifold_score(
            h_np,
            loc_y-1,
            index_type="leaf",
            tree_depth=d-1,
            initial_k=10,
            pca_var=0.95,
            n_permutations=200,
            random_state=0,
        )

        print("Tree Manifold Score (Spearman):", result["score_spearman"])
        print("Permutation p-value:", result["p_value_perm"])
        print("k used:", result["k_used"])
        manifold_score_l.append(result["score_spearman"])
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

        # order = leaves_list(linkage_matrix)

        # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        # (ax1, ax2, ax3, ax4) = axs.flatten()
        # ax1.imshow(distance_matrix[order][:, order], cmap='viridis')
        # ax1.set_title("hidden")
        # ax2.imshow(distance_matrix_states[order][:, order], cmap='viridis')
        # ax2.set_title("tree")
        # ax3.imshow(distance_matrix_inputs[order][:, order], cmap='viridis')
        # ax3.set_title("inputs")
        # ax4.imshow(distance_matrix_outputs[order][:, order], cmap='viridis')
        # ax4.set_title("output")
        # plt.show()

        # t-SNE plot of hidden activity colored by loc_depth and loc_branch
        # from sklearn.manifold import TSNE

        # # Run t-SNE on hidden layer activations
        # tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, len(h_np)-1))
        # h_tsne = tsne.fit_transform(h_np)

        # # Plot t-SNE colored by loc_depth
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
        # scatter = ax1.scatter(h_tsne[:, 0], h_tsne[:, 1], c=loc_depth, cmap='viridis', s=10, alpha=0.8)
        # ax1.set_title("by loc_depth")
        # ax2.scatter(h_tsne[:, 0], h_tsne[:, 1], c=loc_branch, cmap='tab20', s=10, alpha=0.8)
        # ax2.set_title("by loc_branch")
        # ax3.scatter(h_tsne[:, 0], h_tsne[:, 1], c=loc_y_subtrees, cmap='tab20', s=10, alpha=0.8)
        # ax3.set_title("by loc_y_subtrees")
        # plt.tight_layout()
        # plt.show()

        # Compute the similarity (correlation) between all 4 distance matrices and plot using imshow

        from scipy.stats import pearsonr


        # # The order used for all matrices:
        # dm_hidden = distance_matrix[order][:, order]
        # dm_tree = distance_matrix_states[order][:, order]
        # # dm_inputs = distance_matrix_inputs[order][:, order]
        # # dm_outputs = distance_matrix_outputs[order][:, order]
        
        # # Filter all distance matrices
        # dm_hidden = dm_hidden[filter][:, filter]
        # dm_tree = dm_tree[filter][:, filter]
        # dm_inputs = dm_inputs[filter][:, filter]
        # dm_outputs = dm_outputs[filter][:, filter]

        # Since distance matrices are symmetric with zeros on diagonal, ignore diagonal and duplicate entries
        def extract_upper_triangular_values(mat):
            return mat[np.triu_indices_from(mat, k=1)]

        mat_names = ['Hidden', 'Tree', 'Inputs', 'Outputs']
        matrices = [
            distance_matrix,
            distance_matrix_states,
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

        sim_l.append(similarity[0,1])
        sim_mat_l.append(similarity)
        matrices_l.append(matrices)
        loss_l_l.append(loss_l)
        accuracy_l_l.append(accuracy_l)
        PR_l.append(abs(calc_PR(h_np)))
        r2_depth_l.append(r2_depth)
        r2_branch_l.append(r2_branch)
        r2_subtrees_l.append(r2_subtrees)
        test_accuracy_l.append(test_accuracy)
        test_loss_l.append(test_loss)

    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(sim_l, marker='o')
    # plt.ylim(0, 1)
    # fig.savefig('test_tree/sim_l.png')
    # plt.show()

    # # inputs_states_sim = [sim_mat[2,1] for sim_mat in sim_mat_l]
    # # inputs_hidden_sim = [sim_mat[0,2] for sim_mat in sim_mat_l]
    # hidden_states_sim = [sim_mat[0,1] for sim_mat in sim_mat_l]

    # fig = plt.figure(figsize=(10, 5))
    # # plt.plot(inputs_states_sim, marker='o', label='Inputs vs States')
    # # plt.plot(inputs_hidden_sim, marker='o', label='Inputs vs Hidden')
    # plt.plot(hidden_states_sim, marker='o', label='Hidden vs States')
    # plt.legend()
    # fig.savefig('test_tree/inputs_states_hidden_sim.png')
    # plt.show()

    # state_distance = distance_matrix_states[np.triu_indices_from(distance_matrix_states, k=1)]
    # hidden_distance = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    # import pandas as pd

    # # Make sure state_distance and hidden_distance are numpy arrays of same length
    # # Group hidden_distance values according to each unique value in state_distance
    # df = pd.DataFrame({'state_distance': state_distance, 'hidden_distance': hidden_distance})

    # # Sort by state_distance for nicer plots (optional)
    # df = df.sort_values('state_distance')

    # unique_states = np.unique(state_distance)
    # data_to_plot = [df[df['state_distance'] == val]['hidden_distance'].values for val in unique_states]

    # fig = plt.figure(figsize=(10, 5))
    # plt.boxplot(data_to_plot, positions=unique_states)
    # plt.xlabel('State Distance')
    # plt.ylabel('Hidden Distance')
    # plt.title('Hidden Distance Distribution for Each State Distance')
    # plt.xticks(unique_states.astype(int), unique_states.astype(int))  
    # fig.savefig('test_tree/hidden_distance_distribution_boxplot.png')
    # plt.show()

    # fig = plt.figure(figsize=(10, 5))
    # for i, matrices in enumerate(matrices_l):
    #     hidden_distance = matrices[0][np.triu_indices_from(matrices[0], k=1)]
    #     state_distance = matrices[1][np.triu_indices_from(matrices[1], k=1)]
    #     # Make sure state_distance and hidden_distance are numpy arrays of same length
    #     # Group hidden_distance values according to each unique value in state_distance
    #     df = pd.DataFrame({'state_distance': state_distance, 'hidden_distance': hidden_distance})

    #     # Sort by state_distance for nicer plots (optional)
    #     df = df.sort_values('state_distance')

    #     unique_states = np.unique(state_distance)
    #     data_to_plot = [df[df['state_distance'] == val]['hidden_distance'].values for val in unique_states]
    #     plt.plot([d.mean() for d in data_to_plot], marker='o', label=f'A={i+1}')
    # plt.legend()
    # fig.savefig('test_tree/hidden_distance_distribution.png')
    # plt.show()

    # fig = plt.figure(figsize=(10, 5))
    # for i, loss_l in enumerate(loss_l_l):
    #     plt.plot(loss_l, label=f'A={i+1}')
    # plt.axhline(1, ls='--', c='k', alpha=0.5)
    # plt.yscale('log')
    # plt.legend()
    # fig.savefig('test_tree/loss.png')
    # plt.show()

    # from scipy.stats import spearmanr

    # # PLOT 1
    # corr_l = []
    # coverage_l = []
    # for i, matrices in enumerate(matrices_l):
    #     hidden_distance = matrices[0][np.triu_indices_from(matrices[0], k=1)]
    #     state_distance = matrices[1][np.triu_indices_from(matrices[1], k=1)]
    #     coverage_l.append((state_distance <= k_l[i]).mean())
    #     corr_l.append(spearmanr(hidden_distance, state_distance).correlation)
    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(coverage_l, corr_l, marker='o')
    # plt.ylim(-0.1, 1)
    # plt.xlim(-0,1.1)
    # fig.savefig('test_tree/coverage_corr.png')
    # plt.show()

    # # PLOT 2
    # corr_l = []
    # for i, matrices in enumerate(matrices_l):
    #     hidden_distance = matrices[0][np.triu_indices_from(matrices[0], k=1)]
    #     state_distance = matrices[1][np.triu_indices_from(matrices[1], k=1)]
    #     # Make sure state_distance and hidden_distance are numpy arrays of same length
    #     # Group hidden_distance values according to each unique value in state_distance
    #     df = pd.DataFrame({'state_distance': state_distance, 'hidden_distance': hidden_distance})

    #     # Sort by state_distance for nicer plots (optional)
    #     df = df.sort_values('state_distance')

    #     unique_states = np.unique(state_distance)
    #     data_to_plot = [df[df['state_distance'] == val]['hidden_distance'].values for val in unique_states]
    #     mean_h_distance = [d.mean() for d in data_to_plot]
    #     corr_l.append(spearmanr(unique_states, mean_h_distance).correlation)
    # fig = plt.figure(figsize=(10, 5))
    # plt.plot(corr_l, marker='o')
    # plt.ylim(-0.1, 1)
    # fig.savefig('test_tree/corr.png')
    # plt.show()

    # plt.figure()
    # plt.plot(PR_l, marker='o')
    # plt.xlabel('k')
    # plt.ylabel('PR')
    # plt.title('PR vs k')
    # plt.show()

    # plt.figure()
    # plt.plot(k_l, r2_depth_l, marker='o', label='depth')
    # plt.plot(k_l, r2_branch_l, marker='o', label='branch')
    # plt.plot(k_l, r2_subtrees_l, marker='o', label='subtrees')
    # plt.legend()
    # plt.xlabel('k')
    # plt.ylabel('r2')
    # plt.title('r2 vs k')
    # plt.show()


    ax1.plot(k_l, [accuracy_l[-1] for accuracy_l in accuracy_l_l], marker='o')
    ax2.plot(k_l, [loss_l[-1] for loss_l in loss_l_l], marker='o')
    ax3.plot(k_l, test_accuracy_l, marker='o', label=f'{var_name}: {var_value}')
    ax4.plot(k_l, test_loss_l, marker='o')

ax2.set_yscale('log')
ax4.set_yscale('log')

ax1.set_ylabel('Train')
ax3.set_ylabel('Test')

ax1.set_ylim(-0.1, 1.1)
ax3.set_ylim(-0.1, 1.1)

ax1.set_title('Accuracy')
ax2.set_title('Loss')
ax3.legend()
plt.show()

plt.plot(k_l, manifold_score_l, marker='o')
plt.xlabel('k')
plt.ylabel('Manifold Score')
plt.title('Manifold Score vs k')
plt.show()