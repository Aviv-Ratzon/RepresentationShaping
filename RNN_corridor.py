
from sklearn.decomposition import PCA
from torch import nn
from umap import UMAP


np.random.seed(0)
torch.manual_seed(0)


class LinearRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bias=False):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.W_ih = nn.ParameterList()
        self.biases = nn.ParameterList() if bias else None
        
        for layer in range(num_layers):
            in_dim = input_size if layer == 0 else hidden_size
            
            self.W_ih.append(
                nn.Parameter(torch.randn(hidden_size, in_dim) * 0.02)
            )
            
            if bias:
                self.biases.append(
                    nn.Parameter(torch.zeros(hidden_size))
                )
            
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

                h[layer] = linear
                input_t = linear  # input to next layer

            hidden_states.append(h[-1])
            outputs.append(self.output_layer(h[-1]))
        
        hidden_states = torch.stack(hidden_states, dim=1)  # (B, T, H)
        outputs = torch.stack(outputs, dim=1)  # (B, T, H)
        
        return outputs, hidden_states


from copy import deepcopy
import itertools
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance_matrix
from torch import nn, optim

from utils import one_hot
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import get_r_2, calc_PR


use_gpu = True


class Config:
    def __init__(self, **entries):
        self.gpu_id=2
        self.seed=0
        self.G=0.95
        self.hidden_size=100
        self.learning_rate=0.1
        self.num_epochs=10000
        self.L=5
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
        self.algo_name='SGD'
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
C.linear_net = True
C.split_actions = False
C.learning_rate = 0.0001
C.L=1
C.print_progress = True
C.length_corridors = [20]*1
C.max_move = 1
C.hidden_size = 40
C.num_epochs = 100000
C.algo_name = 'Adam'
C.fixed_output = False
C.loss_fn = nn.CrossEntropyLoss()
C.bias = False

order_k_l = []
pr_k_l = []
accuracy_k_l = []
k_l = np.arange(1, 21)
for k in tqdm(k_l):

    C.k = k

    device = torch.device(f"cuda:{C.gpu_id}" if torch.cuda.is_available() and use_gpu else "cpu")
    torch.manual_seed(C.seed)
    np.random.seed(C.seed)

    actions = np.arange(-1, 2)
    n_actions = len(actions)
    actions_in = [one_hot(i, n_actions) for i in range(n_actions)]

    n_states = C.length_corridors[0]
    states = np.arange(n_states)
    states_in = [one_hot(i, n_states) for i in range(n_states)]

    k = C.k

    X = []
    y = []
    loc_X = []
    loc_y = []
    corridor = []
    action_taken = []
    for a_i, (a, a_in) in enumerate(zip(actions, actions_in)):
        for s in states:
            a_curr = a
            a_in_curr = a_in
            if s + a_curr < 0 or s + a_curr >= n_states:
                continue
            X_seq = [np.concatenate([states_in[s], a_in_curr])]
            s_next = s + a_curr
            y_seq = [states_in[s_next]]
            loc_X_seq = [s]
            loc_y_seq = [s_next]
            action_seq = [a_curr]
            for i in range(k-1):
                if s_next + a_curr < 0 or s_next + a_curr >= n_states:
                    a_curr = 0
                    a_in_curr = actions_in[1]
                X_seq.append(np.concatenate([states_in[s_next]*0, a_in_curr]))
                loc_X_seq.append(s_next)
                s_next = s_next + a_curr
                y_seq.append(states_in[s_next])
                loc_y_seq.append(s_next)
                action_seq.append(a_curr)
            
            X.append(X_seq)
            y.append(y_seq)
            loc_X.append(loc_X_seq)
            loc_y.append(loc_y_seq)
            action_taken.append(action_seq)
    X = np.array(X)
    y = np.array(y)
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

    # Create model
    model = LinearRNN(n_states + n_actions, C.hidden_size, n_states, C.L).to(device)
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
    for epoch in tqdm(range(C.num_epochs)):
        optimizer.zero_grad()
        outputs, hidden = model(X)
        loss = criterion(outputs.view(-1, outputs.size(-1)), y.argmax(-1).view(-1))
        loss.backward()
        optimizer.step()
        loss_l.append(loss.item()/y_var)
        accuracy_l.append((outputs.argmax(dim=-1) == y.argmax(dim=-1)).float().mean().item())
        hidden_l.append([h.cpu().detach().numpy() for h in hidden_states])

    print(f'Loss: {loss.item()/y_var}, Accuracy: {accuracy_l[-1]}')
    # Testing
    with torch.no_grad():
        outputs, hidden_states = model(X)
    # print(criterion(outputs, y).item()/y_var)

    only_first_step = True
    if only_first_step:
        X_np = X.cpu().numpy()[:,0,:]
        y_np = y.cpu().numpy()[:,0,:]
        h_np = hidden_states.detach().cpu().numpy()[:,0,:]
        loc_y_np = loc_y[:,0]
        loc_X_np = loc_X[:,0]
        action_taken_np = action_taken[:,0]
    else:
        X_np = X.cpu().numpy().reshape(-1, X.shape[2])  # Convert to numpy array if X is a torch tensor
        y_np = y.cpu().numpy().reshape(-1, y.shape[2])  # Convert to numpy array if y is a torch tensor
        h_np = hidden_states.detach().cpu().numpy().reshape(-1, hidden_states.shape[-1])  # Convert to numpy array if hidden is a torch tensor
        loc_y_np = loc_y.reshape(-1)
        loc_X_np = loc_X.reshape(-1)
        action_taken_np = action_taken.reshape(-1)
    X_dist = distance_matrix(X_np, X_np)
    y_dist = distance_matrix(y_np, y_np)
    hidden_dist = distance_matrix(h_np, h_np)


    order_k_l.append(get_r_2(PCA(n_components=1).fit_transform(h_np), loc_y_np))
    accuracy_k_l.append(accuracy_l[-1])
    pr_k_l.append(calc_PR(h_np).real)

    # indices = np.argsort(loc_y_np)
    # fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # axs[0,0].set_axis_off(); axs[0,2].set_axis_off()
    # axs[0,1].plot(loss_l)
    # axs[0,1].set_yscale('log')
    # ax2 = axs[0,1].twinx()
    # ax2.plot(accuracy_l, 'r')
    # axs[0,1].set_title("Loss")
    # ax2.axhline(y=1, color='gray', linestyle='--')
    # for var, var_name, ax in zip([X_dist, y_dist, hidden_dist], ['X', 'y', 'hidden'], axs[1]):
    #     ax.imshow(var[indices][:, indices], cmap='viridis')
    #     ax.set_title(f'{var_name} distance matrix')
    # plt.tight_layout()
    # plt.show()

    # pca = PCA().fit(h_np)
    # X_reduced = pca.transform(h_np)
    # fig = plt.figure(figsize=(10, 10))

    # # Add cumulative explained variance ratio in the first row
    # ax1 = fig.add_subplot(3, 3, 2)
    # ax1.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    # ax1.set_xlabel('Number of Components')
    # ax1.set_ylabel('Cumulative Explained Variance')
    # ax1.set_title('Cumulative Explained Variance Ratio')

    # # Add scatter plots in the second row
    # for i in range(3):
    #     for j, c in enumerate([loc_y_np, action_taken_np]):
    #         ax = fig.add_subplot(3, 3, i + 4 + j*3)
    #         # c = loc_y
    #         # c = action_taken[inds]
    #         s = ax.scatter(X_reduced[:, i], X_reduced[:, i+1], c=c, cmap='coolwarm', alpha=0.7)
    #         ax.set_xlabel(f'Component {i+1} ({pca.explained_variance_ratio_[i]*100:.2f}%)')
    #         ax.set_ylabel(f'Component {i+2} ({pca.explained_variance_ratio_[i+1]*100:.2f}%)'),
    #         ax.axis('equal')
    #         fig.colorbar(s, ax=ax)

    # plt.tight_layout()
    # plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(k_l, accuracy_k_l, marker='o')
axs[0].set_title('Accuracy')
axs[0].set_ylim(-0.1, 1.1)
axs[1].plot(k_l, order_k_l, marker='o')
axs[1].set_title('Order')
axs[2].plot(k_l, pr_k_l, marker='o')
axs[2].set_title('PR')
[ax.set_xlabel('k') for ax in axs]
plt.show()