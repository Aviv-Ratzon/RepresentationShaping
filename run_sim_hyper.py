from copy import deepcopy
import itertools
from sklearn.decomposition import PCA
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn, optim

from model import DNN
from utils import one_hot, flatten_list
from tqdm import tqdm


use_gpu = True



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


class Config:
    def __init__(self, **entries):
        # Meta
        self.gpu_id=2
        self.seed=0
        self.print_progress=False

        # Data
        self.one_hot_actions=True
        self.one_hot_inputs=True
        self.allow_backwards=True
        self.whiten_data = False
        self.split_actions=False
        self.egocentric_movement=False
        self.length_corridors=[30, 30]
        self.max_move= 15
        self.min_move=0
        self.input_size=100
        self.corridor_dim = 1
        self.input_smoothing = 0

        # Model
        self.sig_h_2 = None
        self.bias = False
        self.fixed_output=False
        self.linear_net=False
        self.G=0.95
        self.hidden_size=100
        self.L=8

        # Training
        self.early_stopping=False
        self.learning_rate=0.00001
        self.num_epochs=10000
        self.algo_name='ADAM'
        self.loss_fn=nn.CrossEntropyLoss()
        self.lambda_reg = 0

class action_handler:
    def __init__(self, C):
        n_cors = len(C.length_corridors)
        n_branches = C.corridor_dim
        cor_len = C.length_corridors[0]
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

def recursive_indexing(l, indexes):
    """
    Recursively index a list of lists.
    """
    if len(indexes) == 1:
        return l[indexes[0]]
    else:
        return recursive_indexing(l[indexes[0]], indexes[1:])


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
        

def create_data(C):
    n_cors = len(C.length_corridors)
    cor_dim = C.corridor_dim
    cor_len = C.length_corridors[0]
    N_inputs = sum([length**cor_dim for length in C.length_corridors])
    # actions = np.concatenate([np.arange(-C.max_move, -C.min_move+1), np.arange(C.min_move, C.max_move + 1)])
    # actions = np.unique(actions)
    # if C.allow_backwards:
    #     run_actions = actions
    # else:
    #     run_actions = actions[actions >= 0]
    # action_id = np.array(list(itertools.product(range(cor_dim), actions, range(n_cors*int(C.split_actions))))).T
    # n_actions = action_id.shape[0]
    # if C.one_hot_actions:
    #     actions_in = [one_hot(i, n_actions) for i in range(n_actions)]
    # else:
    #     actions_in = [np.random.normal(0, 1, size=n_actions) for i in range(n_actions)]
    action_h = action_handler(C)
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


def run_sim(C: Config):
    device = torch.device(f"cuda:{C.gpu_id}" if torch.cuda.is_available() and use_gpu else "cpu")
    torch.manual_seed(C.seed)
    np.random.seed(C.seed)
    loss_thresh = 0.05 if not C.one_hot_inputs else 0.01
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    if C.sig_h_2 and C.print_progress:
        C.G = (C.sig_h_2*(X.shape[1]+C.hidden_size)/(2*X.shape[1]*X.var()))**(1/(2*C.L))
        print(f'Changed G to {C.G} to get sig_h_2 = {C.sig_h_2}')
    # Create model
    model = DNN(input_size + n_actions, C.hidden_size, output_size, C.L, C.fixed_output, C.linear_net, C.G, C.bias).to(device)
    initial_weights = deepcopy(model.state_dict())
    with torch.no_grad():
        outputs, hidden_states = model(X)
        if C.print_progress:
            print(f'Sig_2 of last hidden: {hidden_states[-1].var().item()}')

    # Loss function and optimizer
    criterion = C.loss_fn
    algo = optim.SGD if C.algo_name == 'SGD' else optim.Adam
    optimizer = algo(model.parameters(), lr=C.learning_rate, weight_decay=C.lambda_reg)

    y_var = y.var().cpu().item() if isinstance(criterion, nn.MSELoss) else 1
    # Training loop
    loss_l = []
    accuracy_l = []
    hidden_l = []
    sample_inds = np.unique(np.linspace(0, C.num_epochs-1, 1000).astype(int))
    for epoch in tqdm(range(C.num_epochs)) if C.print_progress else range(C.num_epochs):
        optimizer.zero_grad()
        outputs, hidden_states = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_l.append(loss.item()/y_var)
        if C.one_hot_inputs:
            accuracy_l.append((outputs.argmax(dim=1) == y.argmax(dim=1)).float().mean().item())
            if accuracy_l[-1] == 1 and C.early_stopping:
                # print('perfect accuracy reached, stopping')
                break
        else:
            accuracy_l.append(0)
        if loss_l[-1] < loss_thresh and C.early_stopping:
            # print('loss threshold reached, stopping')
            break
        # if (epoch + 1) % int(C.num_epochs/10) == 0 and C.print_progress:
        #     print(f"Epoch {epoch + 1}/{C.num_epochs}, Loss: {loss_l[-1]:.4f}")
        if epoch in sample_inds:
            hidden_l.append([h.cpu().detach().numpy() for h in hidden_states])

    # Testing
    with torch.no_grad():
        outputs, hidden_states = model(X)
    # print(criterion(outputs, y).item()/y_var)


    return X, y, corridor, loc_X.squeeze(), loc_y.squeeze(), action_taken, hidden_states, loss_l, accuracy_l, outputs.cpu().numpy(), hidden_l, model.state_dict(), initial_weights