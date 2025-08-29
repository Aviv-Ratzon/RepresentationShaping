import itertools
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter


def one_hot(x, num_classes):
    return np.eye(num_classes)[x]


def recursive_indexing(l, indexes):
    """
    Recursively index a list of lists.
    """
    if len(indexes) == 1:
        return l[indexes[0]]
    else:
        return recursive_indexing(l[indexes[0]], indexes[1:])


class action_handler:
    def __init__(self, C):
        n_cors = len(C.length_corridors)
        cor_dim = C.corridor_dim
        actions = np.concatenate([np.arange(-C.max_move, -C.min_move + 1), np.arange(C.min_move, C.max_move + 1)])
        actions = np.unique(actions)
        if C.allow_backwards:
            run_actions = actions
        else:
            run_actions = actions[actions >= 0]
        action_id = np.array(list(itertools.product(actions, range(cor_dim), range(1 + (n_cors-1) * int(C.split_actions))))).T
        n_actions = action_id.shape[1]
        if C.one_hot_actions:
            actions_in = [one_hot(i, n_actions) for i in range(n_actions)]
        else:
            actions_in = [np.random.normal(0, 1, size=n_actions) for i in range(n_actions)]
        self.actions_in =  actions_in
        self.actions_id = action_id
        self.n_actions = n_actions
        self.run_actions = run_actions

    def __call__(self, dim, cor_num, action, split_actions):
        ind = np.where((self.actions_id[0] == action) &
                       (self.actions_id[1] == dim) &
                       (self.actions_id[2] == cor_num*split_actions))[0]
        if len(ind) != 1:
            raise ValueError(f"Action {action} found {len(ind)} times in action_id")
        return self.actions_in[ind[0]]


class TreeNode:
    def __init__(self, index, parent=None):
        self.index = index
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def one_hot(self, total_nodes):
        vec = np.zeros(total_nodes)
        vec[self.index] = 1
        return vec


class FullTree:
    def __init__(self, depth, num_branches):
        self.depth = depth
        self.num_branches = num_branches
        self.nodes = []
        self.root = self._build_tree()

    def _build_tree(self):
        def create_node(index, parent, current_depth):
            node = TreeNode(index, parent)
            self.nodes.append(node)
            if current_depth < self.depth - 1:
                for _ in range(self.num_branches):
                    child_index = len(self.nodes)
                    child = create_node(child_index, node, current_depth + 1)
                    node.add_child(child)
            return node

        return create_node(0, None, 0)

    def get_node_by_index(self, index):
        return self.nodes[index]

    def traverse(self, start_index, actions):
        current = self.get_node_by_index(start_index)
        for action in actions:
            if action == -1:
                if current.parent is None:
                    return None
                current = current.parent
            elif isinstance(action, int) and 0 <= action < len(current.children):
                current = current.children[action]
            else:
                return None
        return current

    def get_one_hot(self, index):
        return self.nodes[index].one_hot(len(self.nodes))


class action_handler_hyper:
    def __init__(self, C):
        n_cors = len(C.length_corridors)
        n_branches = C.corridor_dim
        actions = np.concatenate([np.arange(-C.max_move, -C.min_move + 1), np.arange(C.min_move, C.max_move + 1)])
        actions = np.unique(actions)
        if C.allow_backwards:
            run_actions = actions
        else:
            run_actions = actions[actions >= 0]
        action_id = np.array(list(itertools.product(actions, range(n_branches), range(1 + (n_cors-1) * int(C.split_actions))))).T
        action_id = action_id[:, ~((action_id[0] <= 0) & (action_id[1] > 0))]
        n_actions = action_id.shape[1]
        if C.one_hot_actions:
            actions_in = [one_hot(i, n_actions) for i in range(n_actions)]
        else:
            actions_in = [np.random.normal(0, 1, size=n_actions) for i in range(n_actions)]
        self.actions_in =  actions_in
        self.actions_id = action_id
        self.n_actions = n_actions
        self.run_actions = run_actions

    def __call__(self, dim, cor_num, action, split_actions):
        ind = np.where((self.actions_id[0] == action) &
                       (self.actions_id[1] == dim) &
                       (self.actions_id[2] == cor_num*split_actions))[0]
        if len(ind) != 1:
            return None
        return self.actions_in[ind[0]]


class location_handler:
    def __init__(self, C):
        self.n_cors = len(C.length_corridors)
        self.cor_len = C.length_corridors[0]
        self.n_branches = C.corridor_dim
        self.N_inputs = sum([2**length-1 for length in C.length_corridors])
        self.pos_tree = FullTree(self.cor_len, self.n_branches)
        
    def get_data(self, index, action, n_branch):
        starting_state = self.pos_tree.get_node_by_index(index)
        if action > 0:
            actions = [n_branch for _ in range(action)]
        elif action == 0:
            actions = []
        else:
            actions = [-1 for _ in range(-action)]
        next_state = self.pos_tree.traverse(index, actions)
        return starting_state, next_state


def create_data_euclidean(C):
    """
    Original create_data function from run_sim.py for corridor-based data generation.
    """
    n_cors = len(C.length_corridors)
    cor_dim = C.corridor_dim
    cor_len = C.length_corridors[0]
    N_inputs = sum([length**cor_dim for length in C.length_corridors])
    
    action_h = action_handler(C)
    run_actions = action_h.run_actions
    n_actions = action_h.n_actions
    
    if C.one_hot_inputs:
        input_size = N_inputs
        output_size = N_inputs
        # vecs = [[np.eye(input_size)[sum(C.length_corridors[:i]):sum(C.length_corridors[:i+1])] for _ in range(cor_dim)] for i in range(n_cors)]
        vecs = np.eye(input_size).reshape([n_cors] + [cor_len]*cor_dim + [input_size])
    else:
        input_size = C.input_size
        output_size = C.input_size
        vecs = [gaussian_filter(np.random.normal(size=(C.length_corridors[i]*3, C.input_size)),
                                sigma=C.length_corridors[i]*C.input_smoothing)[C.length_corridors[i]:-C.length_corridors[i]] for i in range(n_cors)]
        vecs = [vecs[i] - vecs[i].mean(axis=0) for i in range(n_cors)]
        vecs = [vec / vec.std() for vec in vecs]
    
    positions = list(itertools.product(*[np.arange(cor_len)]*cor_dim))
    X = []
    y = []
    loc_X = []
    loc_y = []
    corridor = []
    action_taken = []
    dim_l = []
    
    for cor, vec in enumerate(vecs):
        for loc in positions:
            for dim in range(cor_dim):
                for a in run_actions:
                    if not C.egocentric_movement:
                        a = a * (1 if cor == 0 else -1)
                    
                    if (loc[dim] + a < 0) or (loc[dim] + a >= C.length_corridors[cor]) or (dim>0 and a==0):
                        continue
                    
                    action_in = action_h(dim, cor, a, int(C.split_actions))
                    v = recursive_indexing(vec, loc)
                    next_loc = list(loc)
                    next_loc[dim] += a
                    
                    if C.mask_states and any(
                        tuple([loc[i] + (step if i == dim else 0) for i in range(len(loc))]) in C.mask_states
                        for step in (range(0, a + 1) if a >= 0 else range(0, a - 1, -1))
                    ):
                        continue
                    
                    v_next = recursive_indexing(vec, next_loc)
                    corridor.append(cor)
                    dim_l.append(dim)
                    loc_X.append(loc)
                    loc_y.append(next_loc)
                    X.append(np.concatenate([v, action_in]))
                    y.append(v_next)
                    action_taken.append(a)
    
    X = np.array(X)
    y = np.array(y)
    if not C.one_hot_inputs:
        X[:, :input_size] /= X[:, :input_size].std()
        y[:, :output_size] /= y[:, :output_size].std()
    
    corridor = np.array(corridor)
    loc_X = np.array(loc_X)
    loc_y = np.array(loc_y)
    action_taken = np.array(action_taken)
    
    if C.egocentric_movement and C.corridor_dim == 1:
        loc_y[corridor == 1] = max(loc_y) - loc_y[corridor == 1]
        loc_y[corridor == 0] = min(loc_y) + loc_y[corridor == 0]

    if C.whiten_data:
        pca = PCA(whiten=True)
        X = pca.fit_transform(X)

    if C.print_progress:
        print(f'Number of samples: {X.shape[0]}')
        print(f'Input dimension: {X.shape[1]}')
        print(f'Output dimension: {y.shape[1]}')
        print(f'Number of actions: {n_actions}\n')
    
    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions


def create_data_hyperbolic(C):
    """
    create_data function from run_sim_hyper.py for hyperbolic tree-based data generation.
    """
    n_cors = len(C.length_corridors)
    cor_dim = C.corridor_dim
    cor_len = C.length_corridors[0]
    N_inputs = sum([length**cor_dim for length in C.length_corridors])
    
    action_h = action_handler_hyper(C)
    pos_h = location_handler(C)
    run_actions = action_h.run_actions
    n_actions = action_h.n_actions
    n_nodes = pos_h.N_inputs
    input_size = n_nodes
    output_size = n_nodes
    
    X = []
    y = []
    loc_X = []
    loc_y = []
    corridor = []
    action_taken = []
    dim_l = []
    cor = 0
    
    for node_index in range(n_nodes):
        for dim in range(cor_dim):
            for a in action_h.run_actions:
                action_in = action_h(dim, cor, a, int(C.split_actions))
                if action_in is None:
                    continue
                starting_state, next_state = pos_h.get_data(node_index, a, dim)
                if next_state is None:
                    continue
                v = starting_state.one_hot(n_nodes)
                next_node_index = next_state.index
                v_next = next_state.one_hot(n_nodes)
                corridor.append(cor)
                dim_l.append(dim)
                loc_X.append(node_index)
                loc_y.append(next_node_index)
                X.append(np.concatenate([v, action_in]))
                y.append(v_next)
                action_taken.append(a)
    
    X = np.array(X)
    y = np.array(y)
    if not C.one_hot_inputs:
        X[:, :input_size] /= X[:, :input_size].std()
        y[:, :output_size] /= y[:, :output_size].std()
    
    corridor = np.array(corridor)
    loc_X = np.array(loc_X)
    loc_y = np.array(loc_y)
    action_taken = np.array(action_taken)
    
    if C.egocentric_movement and C.corridor_dim == 1:
        loc_y[corridor == 1] = max(loc_y) - loc_y[corridor == 1]
        loc_y[corridor == 0] = min(loc_y) + loc_y[corridor == 0]

    if C.whiten_data:
        pca = PCA(whiten=True)
        X = pca.fit_transform(X)

    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions


def create_data_arm(C):
    """
    Create data for a robotic arm with elbow and wrist positions.
    The arm has two segments: shoulder to elbow, and elbow to wrist.
    Latent variables are theta (shoulder angle) and phi (elbow angle).
    """
    # Set default parameters if not provided
    if not hasattr(C, 'arm_length_1'):
        C.arm_length_1 = 1.0  # Length of first segment (shoulder to elbow)
    if not hasattr(C, 'arm_length_2'):
        C.arm_length_2 = 1.0  # Length of second segment (elbow to wrist)
    if not hasattr(C, 'num_samples'):
        C.num_samples = 1000  # Number of samples to generate
    if not hasattr(C, 'angle_range'):
        C.angle_range = np.pi/10  # Range for angle changes (default: π radians)
    if not hasattr(C, 'noise_std'):
        C.noise_std = 0.01  # Standard deviation of noise to add
    
    # Generate random initial angles
    theta_init = np.random.uniform(0, 2*np.pi, C.num_samples)  # Initial shoulder angle
    phi_init = np.random.uniform(0, 2*np.pi, C.num_samples)    # Initial elbow angle
    
    # Generate random angle changes
    delta_theta = np.random.uniform(-C.angle_range, C.max_move, [C.num_samples, C.max_move])
    delta_phi = np.random.uniform(-C.angle_range, C.max_move, [C.num_samples, C.max_move])
    
    # Calculate initial positions
    # Elbow position
    elbow_x_init = C.arm_length_1 * np.cos(theta_init)
    elbow_y_init = C.arm_length_1 * np.sin(theta_init)
    
    # Wrist position
    wrist_x_init = elbow_x_init + C.arm_length_2 * np.cos(theta_init + phi_init)
    wrist_y_init = elbow_y_init + C.arm_length_2 * np.sin(theta_init + phi_init)
    
    # Calculate final positions
    theta_final = theta_init + delta_theta.sum(-1)
    phi_final = phi_init + delta_phi.sum(-1)
    
    # Final elbow position
    elbow_x_final = C.arm_length_1 * np.cos(theta_final)
    elbow_y_final = C.arm_length_1 * np.sin(theta_final)
    
    # Final wrist position
    wrist_x_final = elbow_x_final + C.arm_length_2 * np.cos(theta_final + phi_final)
    wrist_y_final = elbow_y_final + C.arm_length_2 * np.sin(theta_final + phi_final)
    
    # Create input features: [elbow_x, elbow_y, wrist_x, wrist_y, delta_theta, delta_phi]
    X = np.column_stack([
        elbow_x_init, elbow_y_init, 
        wrist_x_init, wrist_y_init, 
        delta_theta, delta_phi
    ])
    
    # Create target features: [elbow_x_final, elbow_y_final, wrist_x_final, wrist_y_final]
    y = np.column_stack([
        elbow_x_final, elbow_y_final, 
        wrist_x_final, wrist_y_final
    ])
    
    # Create dummy arrays for compatibility with existing interface
    corridor = np.zeros(C.num_samples, dtype=int)  # All samples are from "corridor" 0
    loc_X = np.column_stack([elbow_x_init, elbow_y_init, wrist_x_init, wrist_y_init, theta_init, phi_init])
    loc_y = np.column_stack([elbow_x_final, elbow_y_final, wrist_x_final, wrist_y_final, theta_final, phi_final])
    action_taken = np.column_stack([delta_theta, delta_phi])
    dim_l = np.zeros(C.num_samples, dtype=int)  # Dummy dimension labels
    
    # Set sizes for compatibility
    input_size = 4   # 6 features: [elbow_x, elbow_y, wrist_x, wrist_y, delta_theta, delta_phi]
    output_size = 4  # 4 features: [elbow_x, elbow_y, wrist_x, wrist_y]
    n_actions = 2  # Not really applicable for this dataset, but kept for compatibility
    
    if C.print_progress:
        print(f'Arm Dataset:')
        print(f'Number of samples: {X.shape[0]}')
        print(f'Input dimension: {X.shape[1]} (elbow_x, elbow_y, wrist_x, wrist_y, delta_theta, delta_phi)')
        print(f'Output dimension: {y.shape[1]} (elbow_x, elbow_y, wrist_x, wrist_y)')
        print(f'Arm lengths: {C.arm_length_1}, {C.arm_length_2}')
        print(f'Angle range: ±{C.angle_range:.2f} radians')
        print(f'Noise std: {C.noise_std}\n')
    
    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions


# Dictionary mapping data geometry types to their corresponding create_data functions
DATA_GEOMETRY_FUNCTIONS = {
    'euclidean': create_data_euclidean,
    'hyperbolic': create_data_hyperbolic,
    'arm': create_data_arm,
}


def create_data(C):
    """
    Main create_data function that selects the appropriate data generation method
    based on the data_geometry configuration option.
    """
    if not hasattr(C, 'data_geometry'):
        C.data_geometry = 'euclidean'  # Default to euclidean if not specified
    
    if C.data_geometry not in DATA_GEOMETRY_FUNCTIONS:
        raise ValueError(f"Unknown data_geometry: {C.data_geometry}. "
                        f"Available options: {list(DATA_GEOMETRY_FUNCTIONS.keys())}")
    
    return DATA_GEOMETRY_FUNCTIONS[C.data_geometry](C)
