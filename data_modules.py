import itertools
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from sklearn.datasets import fetch_openml


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
        if hasattr(C, 'action_list'):
            actions = C.action_list
        else:
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
                    
                    next_loc = list(loc)
                    next_loc[dim] += a
                    if not C.cyclic_corridors:
                        if (next_loc[dim] < 0) or (next_loc[dim] >= C.length_corridors[cor]) or (dim>0 and a==0):
                            continue
                    else:
                        next_loc[dim] = next_loc[dim] % C.length_corridors[cor]
                    
                    action_in = action_h(dim, cor, a, int(C.split_actions))
                    v = recursive_indexing(vec, loc)
                    
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
    action_taken = np.column_stack([delta_theta.sum(-1), delta_phi.sum(-1)])
    dim_l = np.zeros(C.num_samples, dtype=int)  # Dummy dimension labels
    
    # Set sizes for compatibility
    input_size = 4   # 6 features: [elbow_x, elbow_y, wrist_x, wrist_y, delta_theta, delta_phi]
    output_size = 4  # 4 features: [elbow_x, elbow_y, wrist_x, wrist_y]
    n_actions = 2*C.max_move  # Not really applicable for this dataset, but kept for compatibility
    
    
    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions


def create_data_non_linear_fn(C):
    """
    Create data for a non-linear function learning task.
    X is tuples [f(s), a] where f is a D-dimensional non-linear function parametrized by a scalar s,
    and a is a scalar in the range [-C.max_move, C.max_move].
    y is f(s+a).
    
    This function supports both continuous and piecewise continuous nonlinear functions.
    """
    # Set default parameters if not provided
    if not hasattr(C, 'num_samples'):
        C.num_samples = 1000  # Number of samples to generate
    if not hasattr(C, 'function_dim'):
        C.function_dim = 10  # Dimensionality of the non-linear function f(s)
    if not hasattr(C, 's_range'):
        C.s_range = (-2.0, 2.0)  # Range for the parameter s (expanded range)
    if not hasattr(C, 'discrete_samples'):
        C.discrete_samples = False  # Whether to use discrete sampling
    if not hasattr(C, 'continuous_function'):
        C.continuous_function = True  # Whether to use continuous or piecewise continuous function
    if not hasattr(C, 'discrete_actions'):
        C.discrete_actions = False  # Whether to use discrete actions
    if not hasattr(C, 'action_dist'):
        C.action_dist = 'uniform'  # Whether to use discrete actions
    
    # Generate random parameter values s
    if C.discrete_samples:
        s_values = np.linspace(C.s_range[0], C.s_range[1], C.num_samples)
        actions = np.arange(-C.max_move, C.max_move + 0.0001, np.diff(s_values).mean())
        # Create all combinations of s_values and actions
        s_grid, a_grid = np.meshgrid(s_values, actions, indexing='ij')
        s_values = s_grid.ravel()
        actions = a_grid.ravel()
    else:
        s_values = np.random.uniform(C.s_range[0], C.s_range[1], C.num_samples)
        if C.action_dist == 'uniform':
            actions = np.random.uniform(-C.max_move, C.max_move, C.num_samples)
        elif C.action_dist == 'normal':
            actions = np.random.normal(0, C.max_move, C.num_samples)
        elif C.action_dist == 'lognormal':
            actions = np.random.lognormal(0, C.max_move, C.num_samples)
        elif C.action_dist == 'poisson':
            actions = np.random.poisson(C.max_move, C.num_samples)
        else:
            raise ValueError(f"Invalid action distribution: {C.action_dist}")

    if C.discrete_actions:
        action_idx = np.random.choice(2*C.n_actions + 1, C.num_samples)
        actions = np.linspace(-C.max_move, C.max_move, 2*C.n_actions + 1)[action_idx]
        one_hot_actions = np.eye(2*C.n_actions + 1)[action_idx]
    
    # Remove samples where s_values + actions are outside of s_range
    s_plus_a = s_values + actions
    valid_mask = (s_plus_a >= C.s_range[0]) & (s_plus_a <= C.s_range[1])
    s_values = s_values[valid_mask]
    actions = actions[valid_mask]
    if C.discrete_actions:
        one_hot_actions = one_hot_actions[valid_mask]
        actions_in = one_hot_actions
    else:
        actions_in = actions.reshape(-1, 1)
    
    # Define the improved non-linear function f(s) with continuous/piecewise options
    functions_slopes = np.random.uniform(C.s_range[0]*2, C.s_range[1]*2, [C.n_breakpoints+1, C.function_dim])
    functions_biases = np.random.uniform(C.s_range[0]*2, C.s_range[1]*2, [C.n_breakpoints+1, C.function_dim])
    def non_linear_function(s):
        """
        Generate high-dimensional nonlinear function output.
        
        Args:
            s: Latent variable values
            continuous: If True, use continuous function; if False, use piecewise continuous
            
        Returns:
            Array of shape (len(s), C.function_dim) containing function outputs
        """
        s = np.atleast_1d(s)
        n = len(s)
        
        if C.continuous_function:
            # Continuous nonlinear function: combination of trigonometric and polynomial terms
            f = np.zeros((n, C.function_dim))
            for i in range(C.function_dim):
                # Different frequency and phase for each dimension
                freq = 1 + 0.1 * i
                phase = 0.5 * i
                f[:, i] = (np.sin(freq * s + phase) + 
                          0.5 * np.cos(2 * freq * s + phase) + 
                          0.1 * s**2 + 
                          0.05 * s**3)
        else:
            # Piecewise continuous function
            f = np.zeros((n, C.function_dim))
            for i in range(C.function_dim):
                # Configurable number of breakpoints per dimension, all within s_range
                n_breakpoints = getattr(C, 'n_breakpoints', 1)
                s_min, s_max = C.s_range
                # Generate breakpoints for this dimension, evenly spaced within s_range
                breakpoints = np.linspace(s_min-0.01, s_max+0.01, n_breakpoints + 2)
                bin_indices = np.digitize(s, breakpoints)-1
                # Assign piecewise function based on which interval s falls into
                f[:, i] = functions_slopes[bin_indices, i] * s + functions_biases[bin_indices, i]
        
        return f
    
    # Calculate function values at s
    f_s = non_linear_function(s_values)
    
    # Calculate function values at s + a
    f_s_plus_a = non_linear_function(s_values + actions)
    
    # Create input features: [f(s), a]
    X = np.concatenate([f_s, actions_in], axis=1)
    
    # Create target features: f(s+a)
    y = f_s_plus_a
    
    # Create dummy arrays for compatibility with existing interface
    corridor = np.zeros(len(s_values), dtype=int)  # All samples are from "corridor" 0
    loc_X = np.column_stack([s_values, actions])
    loc_y = np.column_stack([s_values + actions])
    action_taken = actions
    dim_l = np.zeros(len(s_values), dtype=int)  # Dummy dimension labels
    
    # Set sizes for compatibility
    input_size = C.function_dim  # f(s) + action a
    output_size = C.function_dim      # f(s+a)
    n_actions = actions_in.shape[1]   # Number of possible action values
    
    if C.print_progress:
        print(f'Number of samples: {X.shape[0]}')
        print(f'Input dimension: {X.shape[1]}')
        print(f'Output dimension: {y.shape[1]}')
        print(f'Function dimension: {C.function_dim}')
        print(f'Action range: [-{C.max_move}, {C.max_move}]')
        print(f'Parameter s range: {C.s_range}')
        print(f'Function type: {"Continuous" if C.continuous_function else "Piecewise continuous"}')
        print(f'Number of actions: {n_actions}\n')
    
    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions


def create_data_uneven_corridors(C):
    """
    Create data for uneven 2D discrete state corridors.
    
    This module supports two 2D corridors with different widths and lengths.
    Actions are available in both dimensions within the max_move parameter range.
    Actions can be shared or separated between environments based on split_actions.
    
    Configuration parameters:
    - corridor_widths: List of widths for each corridor [width1, width2]
    - corridor_lengths: List of lengths for each corridor [length1, length2]
    - max_move: Maximum movement in any dimension
    - min_move: Minimum movement (default 0)
    - split_actions: Whether actions are shared (False) or separated (True) between corridors
    - allow_backwards: Whether negative movements are allowed
    - one_hot_actions: Whether to use one-hot encoding for actions
    - one_hot_inputs: Whether to use one-hot encoding for states
    - input_size: Size of input vectors (if not one_hot_inputs)
    - input_smoothing: Smoothing parameter for gaussian filter
    - egocentric_movement: Whether movement direction depends on corridor
    - cyclic_corridors: Whether corridors wrap around
    - mask_states: States to exclude from the dataset
    - whiten_data: Whether to apply PCA whitening
    - print_progress: Whether to print progress information
    """
    # Set default parameters if not provided
    if not hasattr(C, 'corridor_widths'):
        C.corridor_widths = [3, 4]  # Default widths for the two corridors
    if not hasattr(C, 'corridor_lengths'):
        C.corridor_lengths = [5, 6]  # Default lengths for the two corridors
    if not hasattr(C, 'min_move'):
        C.min_move = 0
    if not hasattr(C, 'allow_backwards'):
        C.allow_backwards = True
    if not hasattr(C, 'one_hot_actions'):
        C.one_hot_actions = True
    if not hasattr(C, 'one_hot_inputs'):
        C.one_hot_inputs = True
    if not hasattr(C, 'input_size'):
        C.input_size = 10
    if not hasattr(C, 'input_smoothing'):
        C.input_smoothing = 0.1
    if not hasattr(C, 'egocentric_movement'):
        C.egocentric_movement = False
    if not hasattr(C, 'cyclic_corridors'):
        C.cyclic_corridors = False
    if not hasattr(C, 'mask_states'):
        C.mask_states = None
    if not hasattr(C, 'whiten_data'):
        C.whiten_data = False
    if not hasattr(C, 'print_progress'):
        C.print_progress = False
    if not hasattr(C, 'split_actions'):
        C.split_actions = False
    
    # Validate parameters
    if len(C.corridor_widths) != 2:
        raise ValueError("corridor_widths must have exactly 2 elements")
    if len(C.corridor_lengths) != 2:
        raise ValueError("corridor_lengths must have exactly 2 elements")
    
    n_cors = 2  # Two corridors
    cor_dim = 2  # 2D corridors
    
    # Create action handler for uneven corridors
    class UnevenActionHandler:
        def __init__(self, C):
            self.corridor_widths = C.corridor_widths
            self.corridor_lengths = C.corridor_lengths
            self.max_move = C.max_move
            self.min_move = C.min_move
            self.allow_backwards = C.allow_backwards
            self.one_hot_actions = C.one_hot_actions
            self.split_actions = C.split_actions
            
            # Generate actions
            actions = np.concatenate([
                np.arange(-self.max_move, -self.min_move + 1), 
                np.arange(self.min_move, self.max_move + 1)
            ])
            actions = np.unique(actions)
            
            if self.allow_backwards:
                self.run_actions = actions
            else:
                self.run_actions = actions[actions >= 0]
            
            # Create action IDs: (action, dimension, corridor)
            # If split_actions is True, each corridor gets separate action space
            if self.split_actions:
                action_id = np.array(list(itertools.product(
                    actions, range(cor_dim), range(n_cors)
                ))).T
            else:
                # Shared actions across corridors
                action_id = np.array(list(itertools.product(
                    actions, range(cor_dim), [0]  # All corridors use same action space
                ))).T
            
            self.n_actions = action_id.shape[1]
            self.actions_id = action_id
            
            # Generate action encodings
            if self.one_hot_actions:
                self.actions_in = [one_hot(i, self.n_actions) for i in range(self.n_actions)]
            else:
                self.actions_in = [np.random.normal(0, 1, size=self.n_actions) for i in range(self.n_actions)]
        
        def __call__(self, dim, cor_num, action, split_actions):
            if self.split_actions:
                # Actions are separated by corridor
                ind = np.where((self.actions_id[0] == action) &
                               (self.actions_id[1] == dim) &
                               (self.actions_id[2] == cor_num))[0]
            else:
                # Actions are shared across corridors
                ind = np.where((self.actions_id[0] == action) &
                               (self.actions_id[1] == dim) &
                               (self.actions_id[2] == 0))[0]
            
            if len(ind) != 1:
                raise ValueError(f"Action {action} in dim {dim}, corridor {cor_num} found {len(ind)} times")
            return self.actions_in[ind[0]]
    
    action_h = UnevenActionHandler(C)
    run_actions = action_h.run_actions
    n_actions = action_h.n_actions
    
    # Calculate total number of states
    N_inputs = sum([width * length for width, length in zip(C.corridor_widths, C.corridor_lengths)])
    
    # Generate state vectors
    if C.one_hot_inputs:
        input_size = N_inputs
        output_size = N_inputs
        # Create one-hot vectors for each state
        vecs = []
        state_offset = 0
        for i, (width, length) in enumerate(zip(C.corridor_widths, C.corridor_lengths)):
            corridor_states = width * length
            vec = np.eye(N_inputs)[state_offset:state_offset + corridor_states]
            vecs.append(vec.reshape(width, length, N_inputs))
            state_offset += corridor_states
    else:
        input_size = C.input_size
        output_size = C.input_size
        vecs = []
        for i, (width, length) in enumerate(zip(C.corridor_widths, C.corridor_lengths)):
            # Generate smooth random vectors
            vec = gaussian_filter(
                np.random.normal(size=(width * 3, length * 3, C.input_size)),
                sigma=[width * C.input_smoothing, length * C.input_smoothing, 0]
            )[width:-width, length:-length]
            vec = vec - vec.mean(axis=(0, 1), keepdims=True)
            vec = vec / vec.std(axis=(0, 1), keepdims=True)
            vecs.append(vec)
    
    # Generate all possible positions for each corridor
    X = []
    y = []
    loc_X = []
    loc_y = []
    corridor = []
    action_taken = []
    dim_l = []
    
    for cor, (width, length) in enumerate(zip(C.corridor_widths, C.corridor_lengths)):
        vec = vecs[cor]
        
        # Generate all positions in this corridor
        positions = list(itertools.product(range(width), range(length)))
        
        for loc in positions:
            for dim in range(cor_dim):
                for a in run_actions:
                    if not C.egocentric_movement:
                        a = a * (1 if cor == 0 else -1)
                    
                    next_loc = list(loc)
                    next_loc[dim] += a
                    
                    # Check bounds based on corridor dimensions
                    if not C.cyclic_corridors:
                        if (next_loc[dim] < 0) or (next_loc[dim] >= (width if dim == 0 else length)) or (dim > 0 and a == 0):
                            continue
                    else:
                        # Wrap around
                        if dim == 0:  # Width dimension
                            next_loc[dim] = next_loc[dim] % width
                        else:  # Length dimension
                            next_loc[dim] = next_loc[dim] % length
                    
                    # Get action encoding
                    action_in = action_h(dim, cor, a, int(C.split_actions))
                    
                    # Get state vector
                    v = vec[loc[0], loc[1]]
                    
                    # Check for masked states
                    if C.mask_states and any(
                        tuple([loc[i] + (step if i == dim else 0) for i in range(len(loc))]) in C.mask_states
                        for step in (range(0, a + 1) if a >= 0 else range(0, a - 1, -1))
                    ):
                        continue
                    
                    # Get next state vector
                    v_next = vec[next_loc[0], next_loc[1]]
                    
                    # Store data
                    corridor.append(cor)
                    dim_l.append(dim)
                    loc_X.append(loc)
                    loc_y.append(next_loc)
                    X.append(np.concatenate([v, action_in]))
                    y.append(v_next)
                    action_taken.append(a)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Normalize if not using one-hot inputs
    if not C.one_hot_inputs:
        X[:, :input_size] /= X[:, :input_size].std()
        y[:, :output_size] /= y[:, :output_size].std()
    
    corridor = np.array(corridor)
    loc_X = np.array(loc_X)
    loc_y = np.array(loc_y)
    action_taken = np.array(action_taken)
    
    
    # Apply PCA whitening if requested
    if C.whiten_data:
        pca = PCA(whiten=True)
        X = pca.fit_transform(X)
    
    if C.print_progress:
        print(f'Number of samples: {X.shape[0]}')
        print(f'Input dimension: {X.shape[1]}')
        print(f'Output dimension: {y.shape[1]}')
        print(f'Corridor 1: {C.corridor_widths[0]}x{C.corridor_lengths[0]}')
        print(f'Corridor 2: {C.corridor_widths[1]}x{C.corridor_lengths[1]}')
        print(f'Number of actions: {n_actions}')
        print(f'Split actions: {C.split_actions}')
        print(f'Max move: {C.max_move}')
        print()
    
    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions


def create_data_mnist(C):
    """
    Create data for MNIST with discrete actions. Each sample:
    - Observation: image of a digit (flattened)
    - Action: integer in [-C.max_move, C.max_move]
    - Target: one-hot of class (label + action) mod 10
    """
    if not hasattr(C, 'num_samples'):
        C.num_samples = 10000
    if not hasattr(C, 'one_hot_actions'):
        C.one_hot_actions = True
    if not hasattr(C, 'allow_backwards'):
        C.allow_backwards = True
    if not hasattr(C, 'whiten_data'):
        C.whiten_data = False
    if not hasattr(C, 'seed'):
        C.seed = 0

    # Load MNIST (70k samples, 28x28 flattened to 784)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    images = mnist.data.astype(np.float32) / 255.0
    labels = mnist.target.astype(np.int64)

    # Sample subset
    rng = np.random.default_rng(C.seed)
    idx = rng.choice(images.shape[0], size=min(C.num_samples, images.shape[0]), replace=False)
    images = images[idx]
    labels = labels[idx]

    # Build discrete action space and sample actions
    max_move_int = int(C.max_move)
    action_space = np.arange(-max_move_int, max_move_int + 1, 1, dtype=int)
    n_actions = len(action_space)
    action_indices = rng.integers(0, n_actions, size=images.shape[0])
    actions = action_space[action_indices]

    keep_samples = ((actions+labels) >= min(labels)) & ((actions+labels) <= max(labels))
    images = images[keep_samples]
    labels = labels[keep_samples]
    action_indices = action_indices[keep_samples]
    actions = actions[keep_samples]

    # images_samples = np.stack([images[labels==label][0] for label in np.unique(labels)])
    # image_labels = np.unique(labels)

    # actions_in = action_space[:, None] # np.eye(n_actions, dtype=np.float32)
    # X = []
    # y = []
    # labels = []
    # actions = []
    # for label in np.unique(image_labels):
    #     for action, action_in in zip(action_space, actions_in):
    #         next_label = label + action
    #         if next_label in image_labels:
    #             X.append(np.concatenate([images_samples[label], action_in]))
    #             y.append(images_samples[next_label])
    #             labels.append(next_label)
    #             actions.append(action)

    # X = np.array(X)
    # y = np.array(y)
    # labels = np.array(labels)
    # actions = np.array(actions)
    # Encode actions
    if C.one_hot_actions:
        actions_in = np.eye(n_actions, dtype=np.float32)[action_indices]
    else:
        # Random but fixed embedding per action
        rng_embed = np.random.default_rng(0)
        action_embeddings = rng_embed.normal(0, 1, size=(n_actions, n_actions)).astype(np.float32)
        actions_in = action_embeddings[action_indices]

    # Targets: MNIST images from the resulting class (shifted labels mod 10)
    new_labels = labels + actions
    # For each new label, randomly select an image from the dataset with that label
    images_samples = np.stack([images[labels==label][0] for label in np.unique(labels)])
    y = images_samples[new_labels]

    # Inputs: [image, action encoding]
    X = np.concatenate([images, actions_in], axis=1).astype(np.float32)

    # Meta fields matching interface
    corridor = np.zeros(X.shape[0], dtype=int)
    loc_X = np.column_stack([labels, actions])
    loc_y = labels
    action_taken = actions
    dim_l = np.zeros(X.shape[0], dtype=int)
    action_dim = actions_in.shape[1]

    input_size = images.shape[1]  # 784
    output_size = images.shape[1]

    if C.whiten_data:
        pca = PCA(whiten=True)
        X = pca.fit_transform(X)

    if getattr(C, 'print_progress', False):
        print(f'Number of samples: {X.shape[0]}')
        print(f'Input dimension: {X.shape[1]} (image {input_size} + action {n_actions})')
        print(f'Output dimension: {y.shape[1]} (10 classes)')
        print(f'Number of actions: {n_actions}')

    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, action_dim


# 2D Euclidean grid with local neighborhood actions (square/circle)
def create_data_2d_euclidean(C):
    """
    Create data for a 2D Euclidean grid.

    - States live on a 2D grid of size L x L where L = C.length_corridors (int)
    - Actions are displacements (dx, dy) to neighboring locations defined by:
        * square: |dx| <= C.max_move, |dy| <= C.max_move, excluding (0,0)
        * circle: dx^2 + dy^2 <= C.max_move^2, excluding (0,0)
      Controlled by C.action_shape in {'square', 'circle'}
    - If C.cyclic_corridors: moves wrap around modulo L; otherwise moves outside bounds are skipped
    - Inputs are [state_encoding, action_encoding]
    - Targets are next state's encoding
    - Returns module-standard tuple
    """
    # Defaults
    if not hasattr(C, 'length_corridors'):
        C.length_corridors = 10
    if isinstance(C.length_corridors, (list, tuple)):
        L = C.length_corridors[0]
    else:
        L = int(C.length_corridors)
    if not hasattr(C, 'max_move'):
        C.max_move = 1
    if not hasattr(C, 'min_move'):
        C.min_move = 0
    if not hasattr(C, 'action_shape'):
        C.action_shape = 'square'  # or 'circle'
    if not hasattr(C, 'one_hot_actions'):
        C.one_hot_actions = True
    if not hasattr(C, 'one_hot_inputs'):
        C.one_hot_inputs = True
    if not hasattr(C, 'input_size'):
        C.input_size = 16
    if not hasattr(C, 'input_smoothing'):
        C.input_smoothing = 0.1
    if not hasattr(C, 'cyclic_corridors'):
        C.cyclic_corridors = False
    if not hasattr(C, 'mask_states'):
        C.mask_states = None
    if not hasattr(C, 'whiten_data'):
        C.whiten_data = False
    if not hasattr(C, 'print_progress'):
        C.print_progress = False

    # Build action set
    deltas = []
    for dx in range(-C.max_move, C.max_move + 1):
        for dy in range(-C.max_move, C.max_move + 1):
            if C.action_shape == 'circle':
                if np.sqrt(dx**2 + dy**2) <= C.max_move:
                    deltas.append((dx, dy))
            else:  # square
                deltas.append((dx, dy))
    # Enforce min_move if specified (>0 excludes small steps)
    if C.min_move > 0:
        deltas = [(dx, dy) for (dx, dy) in deltas if (abs(dx) >= C.min_move or abs(dy) >= C.min_move)]

    action_space = np.array(deltas, dtype=int)
    n_actions = action_space.shape[0]

    # State encodings
    N_inputs = L * L
    if C.one_hot_inputs:
        input_size = N_inputs
        output_size = N_inputs
        # One-hot per state index i = x*L + y
        # We will generate on-the-fly via indexing; no huge matrix necessary
        def state_vec(x, y):
            idx = x * L + y
            v = np.zeros(N_inputs, dtype=np.float32)
            v[idx] = 1.0
            return v
    else:
        input_size = C.input_size
        output_size = C.input_size
        grid = gaussian_filter(
            np.random.normal(size=(L * 3, L * 3, C.input_size)),
            sigma=[L * C.input_smoothing, L * C.input_smoothing, 0]
        )[L:-L, L:-L]
        grid = grid - grid.mean(axis=(0, 1), keepdims=True)
        grid = grid / grid.std(axis=(0, 1), keepdims=True)
        def state_vec(x, y):
            return grid[x, y].astype(np.float32)

    # Action encodings
    if C.one_hot_actions:
        actions_in = np.eye(n_actions, dtype=np.float32)
    else:
        rng_embed = np.random.default_rng(0)
        actions_in = rng_embed.normal(0, 1, size=(n_actions, n_actions)).astype(np.float32)

    # Generate dataset over all states and actions
    X = []
    y = []
    loc_X = []
    loc_y = []
    corridor = []
    action_taken = []
    dim_l = []  # Not used in 2D displacement setting; keep zeros for interface

    for x in range(L):
        for y_pos in range(L):
            # Skip masked states if provided
            if C.mask_states and (x, y_pos) in C.mask_states:
                continue
            v = state_vec(x, y_pos)
            for a_idx, (dx, dy) in enumerate(action_space):
                nx = x + dx
                ny = y_pos + dy
                if C.cyclic_corridors:
                    nx %= L
                    ny %= L
                else:
                    if nx < 0 or nx >= L or ny < 0 or ny >= L:
                        continue
                # Skip transitions to masked states
                if C.mask_states and (nx, ny) in C.mask_states:
                    continue
                v_next = state_vec(nx, ny)

                corridor.append(0)
                dim_l.append(0)
                loc_X.append((x, y_pos))
                loc_y.append((nx, ny))
                X.append(np.concatenate([v, actions_in[a_idx]], axis=0))
                y.append(v_next)
                action_taken.append((dx, dy))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    corridor = np.array(corridor, dtype=int)
    loc_X = np.array(loc_X)
    loc_y = np.array(loc_y)
    action_taken = np.array(action_taken)
    dim_l = np.array(dim_l, dtype=int)

    if C.whiten_data:
        pca = PCA(whiten=True)
        X = pca.fit_transform(X)

    if C.print_progress:
        shape = 'circle' if C.action_shape == 'circle' else 'square'
        print(f'2D Euclidean grid L={L}, action_shape={shape}, max_move={C.max_move}')
        print(f'Number of actions: {n_actions}, number of samples: {X.shape[0]}')
        print(f'Input dim: {X.shape[1]} (state {input_size} + action {n_actions if C.one_hot_actions else n_actions})')

    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions


def create_data_random_walk(C):
    """
    Create data for random walk with reflective boundary conditions.
    
    Each data point:
    - Starts from a random position s in [0, C.length_corridors[0]-1]
    - Performs a random walk for C.max_action steps
    - Uses reflective boundary conditions (bounces off walls)
    - Target is the final state after the random walk
    
    Configuration parameters:
    - n_samples: Number of samples to generate (default: 1000)
    - length_corridors: List with first element being number of states (default: [10])
    - max_action: Number of steps in random walk (default: 5)
    - one_hot_inputs: Whether to use one-hot encoding for states (default: True)
    - input_size: Size of input vectors if not one_hot_inputs (default: 10)
    - whiten_data: Whether to apply PCA whitening (default: False)
    - print_progress: Whether to print progress information (default: False)
    """
    # Set default parameters if not provided
    if not hasattr(C, 'n_samples'):
        C.n_samples = 1000
    if not hasattr(C, 'length_corridors'):
        C.length_corridors = [10]
    if not hasattr(C, 'max_move'):
        C.max_move = 5
    if not hasattr(C, 'one_hot_inputs'):
        C.one_hot_inputs = True
    if not hasattr(C, 'input_size'):
        C.input_size = 10
    if not hasattr(C, 'whiten_data'):
        C.whiten_data = False
    if not hasattr(C, 'print_progress'):
        C.print_progress = False
    if not hasattr(C, 'seed'):
        C.seed = 0
    
    # Set random seed
    np.random.seed(C.seed)
    
    n_states = C.length_corridors[0]
    n_samples = C.n_samples
    max_steps = C.max_move
    
    # Generate random starting positions
    start_positions = np.random.randint(0, n_states, n_samples)
    
    # Generate random walks
    X = []
    y = []
    loc_X = []
    loc_y = []
    corridor = []
    action_taken = []
    dim_l = []
    
    for i in range(n_samples):
        current_pos = start_positions[i]
        start_pos = current_pos
        
        # Perform random walk for max_steps
        for step in range(max_steps):
            # Random step: -1 (left) or +1 (right) with equal probability
            step_direction = np.random.choice([-1, 0, 1])
            next_pos = current_pos + step_direction
            
            # Apply reflective boundary conditions
            if next_pos < 0:
                next_pos = 0  # Reflect off left boundary
            elif next_pos >= n_states:
                next_pos = n_states - 1  # Reflect off right boundary
            
            current_pos = next_pos
        
        # After completing the random walk, create one data point
        # Input: [starting_state, action_sequence] where action_sequence is the sequence of steps
        # Target: final_state after the random walk
        
        if C.one_hot_inputs:
            # One-hot encoding for states
            start_state_vec = np.zeros(n_states)
            start_state_vec[start_pos] = 1
            final_state_vec = np.zeros(n_states)
            final_state_vec[current_pos] = 1
        else:
            # Random vector encoding for states
            start_state_vec = np.random.normal(0, 1, C.input_size)
            final_state_vec = np.random.normal(0, 1, C.input_size)
        
        # Action is the sequence of steps taken (encoded as a single value for simplicity)
        # We'll use the net displacement as the action
        net_displacement = current_pos - start_pos
        # One-hot encode net_displacement in range [-(n_states-1), ..., 0, ..., n_states-1]
        action_dim = 2 * (n_states - 1) + 1
        action_offset = n_states - 1  # So that -max maps to 0, 0 maps to n_states-1, +max maps to 2*(n_states-1)
        action_vec = np.zeros(action_dim)
        action_vec[net_displacement + action_offset] = 1
        
        X.append(np.concatenate([start_state_vec, action_vec]))
        y.append(final_state_vec)
        corridor.append(0)  # All samples from corridor 0
        loc_X.append(start_pos)
        loc_y.append(current_pos)
        action_taken.append(net_displacement)
        dim_l.append(0)  # Not applicable for 1D random walk
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    corridor = np.array(corridor, dtype=int)
    loc_X = np.array(loc_X)
    loc_y = np.array(loc_y)
    action_taken = np.array(action_taken)
    dim_l = np.array(dim_l, dtype=int)
    
    # Set sizes
    if C.one_hot_inputs:
        input_size = n_states
        output_size = n_states
    else:
        input_size = C.input_size
        output_size = C.input_size
    
    n_actions = action_dim
    
    # Apply PCA whitening if requested
    if C.whiten_data:
        pca = PCA(whiten=True)
        X = pca.fit_transform(X)
    
    if C.print_progress:
        print(f'Random walk data generation:')
        print(f'Number of samples: {X.shape[0]}')
        print(f'Number of states: {n_states}')
        print(f'Steps per walk: {max_steps}')
        print(f'Input dimension: {X.shape[1]} (state {input_size} + action 1)')
        print(f'Output dimension: {y.shape[1]}')
        print(f'Number of actions: {n_actions}')
        print()
    
    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions


def create_data_back_and_forth(C):
    """
    Create data for random walk with reflective boundary conditions.
    
    Each data point:
    - Starts from a random position s in [0, C.length_corridors[0]-1]
    - Performs a random walk for C.max_action steps
    - Uses reflective boundary conditions (bounces off walls)
    - Target is the final state after the random walk
    
    Configuration parameters:
    - n_samples: Number of samples to generate (default: 1000)
    - length_corridors: List with first element being number of states (default: [10])
    - max_action: Number of steps in random walk (default: 5)
    - one_hot_inputs: Whether to use one-hot encoding for states (default: True)
    - input_size: Size of input vectors if not one_hot_inputs (default: 10)
    - whiten_data: Whether to apply PCA whitening (default: False)
    - print_progress: Whether to print progress information (default: False)
    """
    # Set default parameters if not provided
    if not hasattr(C, 'n_samples'):
        C.n_samples = 1000
    if not hasattr(C, 'length_corridors'):
        C.length_corridors = [10]
    if not hasattr(C, 'max_move'):
        C.max_move = 5
    if not hasattr(C, 'one_hot_inputs'):
        C.one_hot_inputs = True
    if not hasattr(C, 'input_size'):
        C.input_size = 10
    if not hasattr(C, 'whiten_data'):
        C.whiten_data = False
    if not hasattr(C, 'print_progress'):
        C.print_progress = False
    if not hasattr(C, 'seed'):
        C.seed = 0
    
    # Set random seed
    np.random.seed(C.seed)
    
    n_states = C.length_corridors[0]
    n_samples = C.n_samples
    max_steps = C.max_move
    
    # Generate random starting positions
    start_positions = np.random.randint(0, n_states, n_samples)
    
    # Generate random walks
    X = []
    y = []
    loc_X = []
    loc_y = []
    corridor = []
    action_taken = []
    dim_l = []
    
    for i in range(n_samples):
        current_pos = start_positions[i]
        start_pos = current_pos
        
        # Perform random walk for max_steps
        for a in [-1, 0, 1]:
            for step in range(max_steps):
                # Random step: -1 (left) or +1 (right) with equal probability
                next_pos = current_pos + a
                
                # Apply reflective boundary conditions
                if next_pos < 0:
                    next_pos = 1  # Reflect off left boundary
                    a = -a
                elif next_pos >= n_states:
                    next_pos = n_states - 2  # Reflect off right boundary
                    a = -a
                
                current_pos = next_pos
                if a == 0:
                    break
        
        # After completing the random walk, create one data point
        # Input: [starting_state, action_sequence] where action_sequence is the sequence of steps
        # Target: final_state after the random walk
        
        if C.one_hot_inputs:
            # One-hot encoding for states
            start_state_vec = np.zeros(n_states)
            start_state_vec[start_pos] = 1
            final_state_vec = np.zeros(n_states)
            final_state_vec[current_pos] = 1
        else:
            # Random vector encoding for states
            start_state_vec = np.random.normal(0, 1, C.input_size)
            final_state_vec = np.random.normal(0, 1, C.input_size)
        
        # Action is the sequence of steps taken (encoded as a single value for simplicity)
        # We'll use the net displacement as the action
        net_displacement = current_pos - start_pos
        # One-hot encode net_displacement in range [-(n_states-1), ..., 0, ..., n_states-1]
        action_dim = 2 * (n_states - 1) + 1
        action_offset = n_states - 1  # So that -max maps to 0, 0 maps to n_states-1, +max maps to 2*(n_states-1)
        action_vec = np.zeros(action_dim)
        action_vec[net_displacement + action_offset] = 1
        
        X.append(np.concatenate([start_state_vec, action_vec]))
        y.append(final_state_vec)
        corridor.append(0)  # All samples from corridor 0
        loc_X.append(start_pos)
        loc_y.append(current_pos)
        action_taken.append(net_displacement)
        dim_l.append(0)  # Not applicable for 1D random walk
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    corridor = np.array(corridor, dtype=int)
    loc_X = np.array(loc_X)
    loc_y = np.array(loc_y)
    action_taken = np.array(action_taken)
    dim_l = np.array(dim_l, dtype=int)
    
    # Set sizes
    if C.one_hot_inputs:
        input_size = n_states
        output_size = n_states
    else:
        input_size = C.input_size
        output_size = C.input_size
    
    n_actions = action_dim
    
    # Apply PCA whitening if requested
    if C.whiten_data:
        pca = PCA(whiten=True)
        X = pca.fit_transform(X)
    
    if C.print_progress:
        print(f'Random walk data generation:')
        print(f'Number of samples: {X.shape[0]}')
        print(f'Number of states: {n_states}')
        print(f'Steps per walk: {max_steps}')
        print(f'Input dimension: {X.shape[1]} (state {input_size} + action 1)')
        print(f'Output dimension: {y.shape[1]}')
        print(f'Number of actions: {n_actions}')
        print()
    
    return X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions


# Dictionary mapping data geometry types to their corresponding create_data functions
DATA_GEOMETRY_FUNCTIONS = {
    'euclidean': create_data_euclidean,
    'hyperbolic': create_data_hyperbolic,
    'arm': create_data_arm,
    'non_linear_fn': create_data_non_linear_fn,
    'uneven_corridors': create_data_uneven_corridors,
    'MNIST': create_data_mnist,
    'mnist': create_data_mnist,
    '2d_euclidean': create_data_2d_euclidean,
    'random_walk': create_data_random_walk,
    'back_and_forth': create_data_back_and_forth,
}


def create_data(C):
    np.random.seed(C.seed)
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
