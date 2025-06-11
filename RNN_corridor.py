
from sklearn.decomposition import PCA
from torch import nn
from umap import UMAP

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, fixed_output=False, linear_net=False, G=1, bias=False):
        super(DNN, self).__init__()
        self.num_layers = num_layers
        self.bias = bias

        # Define layers
        self.input_layer = nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='relu', batch_first=True, bias=bias)
        if num_layers > 1:
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=num_layers - 1, nonlinearity='relu', batch_first=True, bias=bias)
        self.output_layer = nn.Linear(hidden_size, output_size, bias=bias)
        if fixed_output:
            self.output_layer.requires_grad_(False)

        if linear_net:
            self.activation = nn.Identity()
        else:
            self.activation = nn.ReLU()

        # Initialize weights using Xavier with gain 0.1, and set biases to zero
        self.init_weights(fixed_output, G)

    def init_weights(self, fixed_output, G):
        for name, param in self.input_layer.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=G)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        if self.num_layers > 1:
            for name, param in self.rnn.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param, gain=G)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        if fixed_output:
            nn.init.normal_(self.output_layer.weight)
        else:
            nn.init.xavier_normal_(self.output_layer.weight, gain=G)
        if self.bias:
            nn.init.constant_(self.input_layer.bias, 0.0)
            nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):

        # Input layer
        x = self.activation(self.input_layer(x)[0])  # x is of shape (batch_size, seq_len, input_size)
        # hidden_states.append(x)

        # RNN layers
        if self.num_layers > 1:
            x, _ = self.rnn(x)  # Add batch dimension for RNN
        hidden_states = x.clone() # Store the last hidden state of the RNN

        # Output layer
        out = self.output_layer(x)

        return out, hidden_states


from copy import deepcopy
import itertools
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn, optim

from utils import one_hot
from tqdm import tqdm


use_gpu = True


class Config:
    def __init__(self, **entries):
        self.gpu_id=2
        self.seed=0
        self.G=0.95
        self.hidden_size=100
        self.learning_rate=0.00001
        self.num_epochs=10000
        self.L=8
        self.input_size=100
        self.length_corridors=[30, 30]
        self.max_move= 1
        self.min_move= 0
        self.one_hot_actions=True
        self.one_hot_inputs=True
        self.allow_backwards=True
        self.fixed_output=False
        self.egocentric_movement=False
        self.linear_net=False
        self.algo_name='ADAM'
        self.loss_fn=nn.CrossEntropyLoss()
        self.split_actions=False
        self.early_stopping=False
        self.print_progress=False
        self.corridor_dim = 1
        self.input_smoothing = 0
        self.sig_h_2 = None
        self.bias = False
        self.k = 1
        # self.__dict__.update(entries)

C = Config()

C.G = 0.8
C.sig_h_2 = 1e-8
C.linear_net = False
C.split_actions = False
C.learning_rate = 0.0001
C.L=10
C.print_progress = True
C.length_corridors = [10]*1
C.max_move = 1
C.hidden_size = 200
C.num_epochs *= 1
C.algo_name = 'ADAM'
C.fixed_output = False
C.loss_fn = nn.CrossEntropyLoss()
C.bias = False
C.k = 5

device = torch.device(f"cuda:{C.gpu_id}" if torch.cuda.is_available() and use_gpu else "cpu")
torch.manual_seed(C.seed)
np.random.seed(C.seed)
n_cors = len(C.length_corridors)
N_inputs = sum(C.length_corridors)
loss_thresh = 0.05 if not C.one_hot_inputs else 0.01
actions = np.concatenate([np.arange(-C.max_move, -C.min_move+1), np.arange(C.min_move, C.max_move + 1)])
actions = np.unique(actions)
if C.allow_backwards:
    run_actions = actions
else:
    run_actions = actions[actions >= 0]
n_actions = len(actions) + (int(C.split_actions)*(n_cors-1) * len(actions))
if C.one_hot_actions:
    actions_in = [one_hot(i, n_actions) for i in range(n_actions)]
else:
    actions_in = [np.random.normal(0, 1, size=n_actions) for i in range(n_actions)]

if C.one_hot_inputs:
    input_size = N_inputs
    output_size = N_inputs
    vecs = [np.eye(input_size)[sum(C.length_corridors[:i]):sum(C.length_corridors[:i+1])] for i in range(n_cors)]
else:
    input_size = C.input_size
    output_size = C.input_size
    vecs = [gaussian_filter(np.random.normal(size=(C.length_corridors[i]*3, C.input_size)),
                            sigma=C.length_corridors[i]*C.input_smoothing)[C.length_corridors[i]:-C.length_corridors[i]] for i in range(n_cors)]
    vecs = [vecs[i] - vecs[i].mean(axis=0) for i in range(n_cors)]
    vecs = [vec / vec.std() for vec in vecs]
    # L_vec = gaussian_filter(np.random.normal(size=(C.length_corridors[0] * 3, C.input_size)), sigma=0)[C.length_corridors[0]:2 * C.length_corridors[0]]
    # R_vec = gaussian_filter(np.random.normal(size=(C.length_corridors[1] * 3, C.input_size)), sigma=0)[C.length_corridors[1]:2 * C.length_corridors[1]]
X = []
y = []
loc_X = []
loc_y = []
corridor = []
action_taken = []
for cor, vec in enumerate(vecs):
    for loc, v in enumerate(vec):
        for a_seq in list(itertools.combinations_with_replacement(run_actions, C.k+1)):
            X_curr = []
            y_curr = []
            loc_X_curr = []
            loc_y_curr = []
            save_seq = True
            end_loc = loc
            for seq_i, a in enumerate(a_seq):
                end_loc += a
                a_i = np.where(actions == a)[0][0]
                a_i += (cor * int(C.split_actions) * (n_actions//n_cors))
                if end_loc < 0 or end_loc >= C.length_corridors[cor]:
                    save_seq = False
                    break
                stim = v if seq_i == 0 else v*0
                X_curr.append(np.concatenate([stim, actions_in[a_i]]))
                y_curr.append(vec[end_loc])
                loc_X_curr.append(loc)
                loc_y_curr.append(end_loc)
            if save_seq:
                if len(X_curr) != C.k+1:
                    print(f"Error: X_curr length is {len(X_curr)} expected {C.k+1}")
                X.append(X_curr)
                y.append(y_curr)
                corridor.append(cor)
                loc_X.append(loc_X_curr)
                loc_y.append(loc_y_curr)
                action_taken.append(a_seq)
X = np.array(X)
y = np.array(y)
if not C.one_hot_inputs:
    X[:, :input_size] /= X[:, :input_size].std()
    y[:, :output_size] /= y[:, :output_size].std()
corridor = np.array(corridor)
loc_X = np.array(loc_X)
loc_y = np.array(loc_y)
action_taken = np.array(action_taken)



X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)
# cond = (loc_X>=-min(action_taken))&(loc_X<=max(loc_X) - max(action_taken))
# X = X[cond]
# y = y[cond]
# corridor = corridor[cond]
# loc_X = loc_X[cond]
# loc_y = loc_y[cond]
# action_taken = action_taken[cond]

if C.sig_h_2:
    C.G = (C.sig_h_2*(X.shape[1]+C.hidden_size)/(2*X.shape[1]*X.var()))**(1/(2*C.L))
    print(f'Changed G to {C.G} to get sig_h_2 = {C.sig_h_2}')
# Create model
model = DNN(input_size + n_actions, C.hidden_size, output_size, C.L, C.fixed_output, C.linear_net, C.G, C.bias).to(device)
initial_weights = deepcopy(model.state_dict())
with torch.no_grad():
    outputs, hidden_states = model(X)
    print(f'Sig_2 of last hidden: {hidden_states[-1].var().item()}')

# Loss function and optimizer
criterion = C.loss_fn
algo = optim.SGD if C.algo_name == 'SGD' else optim.Adam
optimizer = algo(model.parameters(), lr=C.learning_rate)

y_var = y.var().cpu() if isinstance(criterion, nn.MSELoss) else 1
# Training loop
loss_l = []
accuracy_l = []
hidden_l = []
sample_inds = np.unique(np.linspace(0, C.num_epochs-1, 1000).astype(int))
for epoch in tqdm(range(C.num_epochs)) if C.print_progress else range(C.num_epochs):
    optimizer.zero_grad()
    outputs, hidden_states = model(X)
    loss = criterion(outputs.view(-1, outputs.size(-1)), y.argmax(-1).view(-1))
    loss.backward()
    optimizer.step()
    loss_l.append(loss.item()/y_var)
    if C.one_hot_inputs:
        accuracy_l.append((outputs.argmax(dim=-1) == y.argmax(dim=-1)).float().mean().item())
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

import matplotlib.pyplot as plt

hidden = hidden_states.detach().cpu().numpy()
X_dist = torch.cdist(X[:,0,:], X[:,0,:]).cpu().numpy()
y_dist = torch.cdist(y[:,0,:], y[:,0,:]).cpu().numpy()
hidden_dist = torch.cdist(hidden_states.detach(), hidden_states.detach()).cpu().numpy()
stay_inds = np.where(action_taken == 0)[0]
loc_y_corridor = loc_y + (corridor * max(loc_y + 1))

X_np = X.cpu().numpy()  # Convert to numpy array if X is a torch tensor
y_np = y.cpu().numpy()  # Convert to numpy array if y is a torch tensor
h_np = hidden  # Convert to numpy array if hidden is a torch tensor

indices = np.lexsort((loc_y, corridor))
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0,0].set_axis_off(); axs[0,2].set_axis_off()
axs[0,1].plot(loss_l)
axs[0,1].set_yscale('log')
axs[0,1].twinx().plot(accuracy_l, 'r')
axs[0,1].set_title("Loss")
for var, var_name, ax in zip([X_dist, y_dist, hidden_dist], ['X', 'y', 'hidden'], axs[1]):
    ax.imshow(var[indices][:, indices], cmap='viridis')
    ax.set_title(f'{var_name} distance matrix')
plt.tight_layout()
plt.show()

pca = PCA().fit(h_np)
X_reduced = pca.transform(h_np)
fig = plt.figure(figsize=(10, 10))

# Add cumulative explained variance ratio in the first row
ax1 = fig.add_subplot(3, 3, 2)
ax1.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
ax1.set_xlabel('Number of Components')
ax1.set_ylabel('Cumulative Explained Variance')
ax1.set_title('Cumulative Explained Variance Ratio')

# Add scatter plots in the second row
for i in range(3):
    for j, c in enumerate([loc_y, loc_X]):
        ax = fig.add_subplot(3, 3, i + 4 + j*3)
        # c = loc_y
        # c = action_taken[inds]
        s = ax.scatter(X_reduced[:, i], X_reduced[:, i+1], c=c, cmap='coolwarm', alpha=0.7)
        ax.set_xlabel(f'Component {i+1}')
        ax.set_ylabel(f'Component {i+2}'),
        ax.axis('equal')
        fig.colorbar(s, ax=ax)

plt.tight_layout()
plt.show()