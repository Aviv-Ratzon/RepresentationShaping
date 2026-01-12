from copy import deepcopy
import itertools
from sklearn.decomposition import PCA
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn, optim

from model import DNN
from tqdm import tqdm
from data_modules import create_data
from utils import calc_PR

# from utils import state_dict_to_theta, theta_to_state_dict



def state_dict_to_theta(model_dict):
    """
    Convert a state dictionary to a flattened parameter vector theta.
    
    Args:
        model_dict: Dictionary containing model parameters
        
    Returns:
        theta: Flattened parameter vector
        shapes: List of original tensor shapes for reconstruction
        sizes: List of tensor sizes for reconstruction
    """
    W_l = [W.clone().detach() for W in model_dict.values()]
    shapes = [W.shape for W in W_l]
    sizes = [W.numel() for W in W_l]
    theta = torch.concatenate([W.reshape(-1) for W in model_dict.values()])
    return theta, shapes, sizes


def theta_to_state_dict(theta, model_dict, shapes=None, sizes=None):
    """
    Convert a flattened parameter vector theta back to a state dictionary.
    
    Args:
        theta: Flattened parameter vector
        model_dict: Original model dictionary (for keys and device)
        shapes: List of tensor shapes (optional, computed if None)
        sizes: List of tensor sizes (optional, computed if None)
        
    Returns:
        new_model_dict: State dictionary with reconstructed parameters
    """
    if shapes is None or sizes is None:
        W_l = [W.clone().detach() for W in model_dict.values()]
        shapes = [W.shape for W in W_l]
        sizes = [W.numel() for W in W_l]
    
    W_l_new = []
    idx = 0
    for shape, size in zip(shapes, sizes):
        W_l_new.append(theta[idx:idx+size].reshape(shape))
        idx += size
    new_model_dict = {k: v for k, v in zip(model_dict.keys(), W_l_new)}
    return new_model_dict

use_gpu = True


class Config:
    def __init__(self, **entries):
        # Meta
        self.gpu_id=0
        self.seed=0
        self.print_progress=False

        # Data
        self.one_hot_actions=True
        self.scalar_actions=False  # If True, encode actions as a single scalar value instead of vector
        self.one_hot_inputs=True
        self.one_hot_outputs=True
        self.allow_backwards=True
        self.whiten_data = False
        self.split_actions=True
        self.egocentric_movement=True
        self.length_corridors=[30]
        self.max_move= 15
        self.min_move=0
        self.input_size=100
        self.output_size=100
        self.corridor_dim = 1
        self.input_smoothing = 0
        self.mask_states = None
        self.data_geometry = 'euclidean'  # Options: 'euclidean', 'hyperbolic'
        self.cyclic_corridors = False

        # Model
        self.sig_h_2 = None
        self.bias = False
        self.fixed_output=False
        self.linear_net=False
        self.activation='ReLU'  # Options: 'ReLU', 'Tanh', 'Sigmoid', 'Identity', etc.
        self.G=0.95
        self.hidden_size=100
        self.L=8

        # Training
        self.early_stopping=False
        self.learning_rate=0.1
        self.num_epochs=10000
        self.algo_name='SGD'
        self.loss_fn=nn.CrossEntropyLoss()
        self.lambda_reg = 0
        self.B = 1
        self.label_noise = 0
        self.isotropic_noise = 0
        self.bias_batch = None
        self.state_dict_path = None
        self.normalize_theta = False
        
        # Numerical precision settings
        self.use_high_precision = True  # Use float64 instead of float32



def train_model(C: Config, X, y, model, action_taken):
    with torch.no_grad():
        outputs, hidden_states = model(X)
        if C.print_progress:
            print(f'Sig_2 of last hidden: {hidden_states[-1].var().item()}')

    # Loss function and optimizer
    criterion = C.loss_fn
    algo = optim.SGD if C.algo_name == 'SGD' else optim.Adam
    optimizer = algo(model.parameters(), lr=C.learning_rate, weight_decay=C.lambda_reg)
    loss_thresh = 0.05 if not C.one_hot_inputs else 0.01

    # Enable higher precision training if configured
    if C.use_high_precision:
        model = model.double()  # Convert to float64 for higher precision
        X = X.double()
        y = y.double()
    
    
    y_var = y.var().cpu().item() if isinstance(criterion, nn.MSELoss) else 1
    # Training loop
    loss_l = []
    accuracy_l = []
    hidden_l = []
    state_dict_l = []
    PR_l = []
    sample_inds = np.unique(np.linspace(0, C.num_epochs-1, 10000).astype(int))
    sample_inds_state_dict = np.unique(np.linspace(0, C.num_epochs-1, 100).astype(int))
    sample_inds_PR = np.unique(np.linspace(0, C.num_epochs-1, 100).astype(int))
    for epoch in tqdm(range(C.num_epochs)) if C.print_progress else range(C.num_epochs):
        if C.B == 1:
            X_batch = X
            y_batch = y
        else:
            if C.bias_batch:
                p = (C.max_move - abs(action_taken) + 1)**C.bias_batch
                p = p / p.sum()
            else:
                p = None
            batch_inds = np.random.choice(X.shape[0], size=int(C.B*X.shape[0]), replace=True, p=p)
            X_batch = X[batch_inds]
            y_batch = y[batch_inds]
        optimizer.zero_grad()
        outputs, hidden_states = model(X_batch)
        if not isinstance(criterion, nn.CrossEntropyLoss):
            y_batch = y_batch + torch.randn_like(y_batch) * C.label_noise
        loss = criterion(outputs, y_batch)
        loss.backward()
        for param in model.parameters():
            param.grad += torch.randn_like(param.grad) * C.isotropic_noise
        optimizer.step()
        # if (epoch + 1) % int(C.num_epochs/10) == 0 and C.print_progress:
        #     print(f"Epoch {epoch + 1}/{C.num_epochs}, Loss: {loss_l[-1]:.4f}")
        
        if C.normalize_theta:
            model_dict = model.state_dict()
            theta = torch.concatenate([W.reshape(-1) for W in model_dict.values()])
            theta_norm = torch.linalg.norm(theta)
            new_model_dict = {k:v*int(C.normalize_theta)/theta_norm for k, v in model_dict.items()}
            model.load_state_dict(new_model_dict)
        

        with torch.no_grad():
            if epoch in sample_inds[::10]:
                outputs, hidden_states = model(X)
                # hidden_l.append([h.cpu().detach().numpy() for h in hidden_states])

            if epoch in sample_inds:
                outputs, hidden_states = model(X)
                loss = criterion(outputs, y)
                loss_l.append(loss.item()/y_var)
                if C.one_hot_inputs:
                    accuracy_l.append((outputs.argmax(dim=1) == y.argmax(dim=1)).float().mean().item())
                    if (accuracy_l[-1] == 1 or loss_l[-1] < loss_thresh) and C.early_stopping:
                        # print('perfect accuracy reached, stopping')
                        break
                else:
                    accuracy_l.append(0)
            
            if epoch in sample_inds_PR:
                outputs, hidden_states = model(X)
                # Compute PR for the last hidden layer
                last_hidden = hidden_states[-1].cpu().detach().numpy()
                PR_value = calc_PR(last_hidden)
                PR_l.append(PR_value)
                
        # if epoch in sample_inds_state_dict:
        #     state_dict_l.append(deepcopy(model.state_dict()))
    
    model.float()
    return loss_l, accuracy_l, hidden_l, state_dict_l, PR_l


def run_sim(C: Config):
    device = torch.device(f"cuda:{C.gpu_id}" if torch.cuda.is_available() and use_gpu else "cpu")
    if C.seed is not None:
        torch.manual_seed(C.seed)
        np.random.seed(C.seed)
    X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    if C.sig_h_2 is not None and C.L > 0:
        C.G = ((C.sig_h_2*(X.shape[1]+C.hidden_size)/(2*X.shape[1]*X.var()))**(1/(2*C.L))).item()
    if C.sig_h_2 and C.print_progress:
        print(f'Changed G to {C.G} to get sig_h_2 = {C.sig_h_2}')
    # Create model
    model = DNN(input_size + n_actions, C.hidden_size, output_size, C.L, C.fixed_output, C.linear_net, C.G, C.bias, C.activation).to(device)
    if C.state_dict_path is not None:
        model.load_state_dict(torch.load(C.state_dict_path))
    initial_weights = deepcopy(model.state_dict())

    loss_l, accuracy_l, hidden_l, state_dict_l, PR_l = train_model(C, X, y, model, action_taken)
    # Testing
    with torch.no_grad():
        outputs, hidden_states = model(X)
    # print(criterion(outputs, y).item()/y_var)


    return X, y, corridor, loc_X.squeeze(), loc_y.squeeze(), action_taken, hidden_states, loss_l, accuracy_l, outputs.cpu().numpy(), hidden_l, model.state_dict(), initial_weights, state_dict_l, PR_l



def run_sim_wrapper(C):
    X, y, corridor, loc_X, loc_y, action_taken, hidden_states, loss_l, accuracy_l, outputs, hidden_l, final_weights, initial_weights, state_dict_l, PR_l = run_sim(C)

    data_dict = {
        'X': X,
        'y': y,
        'corridor': corridor,
        'loc_X': loc_X.squeeze(),
        'loc_y': loc_y.squeeze(),
        'action_taken': action_taken,
        'hidden_states': hidden_states,
        'loss_l': loss_l,
        'accuracy_l': accuracy_l,
        'outputs': outputs,
        'hidden_l': hidden_l,
        'initial_weights': initial_weights,
        'final_weights': final_weights,
        'C': C,
        'state_dict_l': state_dict_l,
        'PR_l': PR_l
    }

    return data_dict