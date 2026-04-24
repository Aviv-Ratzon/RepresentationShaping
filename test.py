from data_modules import create_data
from run_sim import Config
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


C = Config()
C.length_corridors = [20]*1
C.max_move = 10
C.one_hot_inputs = True
C.one_hot_actions = True
C.print_progress = False
C.seed = 0

X, y, corridor, loc_X, loc_y, action_taken, dim_l, input_size, output_size, n_actions = create_data(C)
labels = y.argmax(1)
# Whiten X
X_mean = np.mean(X, axis=0, keepdims=True)
X_std = np.std(X, axis=0, keepdims=True)
X = (X - X_mean) / (X_std + 1e-8)

# Whiten y
y_mean = np.mean(y, axis=0, keepdims=True)
y_std = np.std(y, axis=0, keepdims=True)
y = (y - y_mean) / (y_std + 1e-8)


# Prepare data
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Network definition
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.hidden_layers(x)
        last_hidden = x
        out = self.output_layer(x)
        return out, last_hidden

    def _compute_regularization_loss(self):
        reg_loss = 0.0
        num_params = 0
        for param in self.parameters():
            if param.dim() > 1:
                reg_loss += (param**(2)).sum()
                num_params += param.numel()
        reg_loss = reg_loss / num_params
        return reg_loss

# Model initialization
input_dim = X.shape[1]
hidden_dim = 2048
output_dim = y_tensor.shape[1]

n_layers_l = [1, 2]
labmda_l = [0, 0.001, 0.1]
fig_pca, axs_pca = plt.subplots(len(n_layers_l), len(labmda_l), figsize=(10, 10))
fig_loss, axs_loss = plt.subplots(len(n_layers_l), len(labmda_l), figsize=(10, 10), sharey=True)
axs_pca[0,0].set_title("Full batch")
axs_loss[0,0].set_title("Full batch")
axs_pca[0,1].set_title("10% batch")
axs_loss[0,1].set_title("10% batch")
for i_layer, n_layers in enumerate(n_layers_l):
    axs_pca[i_layer,0].set_ylabel(f"Layers={n_layers}")
    axs_loss[i_layer,0].set_ylabel(f"Layers={n_layers}")
    for i_lambda, lambda_ in enumerate(labmda_l):
        axs_curr_pca = axs_pca[i_layer, i_lambda]
        axs_curr_loss = axs_loss[i_layer, i_lambda]
        print(f"Training model with {n_layers} layers and batch size factor {lambda_}")
        model = SimpleMLP(input_dim, hidden_dim, output_dim, n_layers=n_layers)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        # Training loop
        loss_l = []
        epochs = 1000
        bsz = X.shape[0]
        for epoch in tqdm(range(epochs), desc="Training"):
            batch_inds = np.random.choice(X.shape[0], size=bsz, replace=False)
            X_batch = X_tensor[batch_inds]
            y_batch = y_tensor[batch_inds]
            model.train()
            optimizer.zero_grad()
            out, _ = model(X_batch)
            mse_loss = criterion(out, y_batch)
            reg_loss = lambda_ * model._compute_regularization_loss()
            loss = mse_loss + reg_loss
            loss.backward()
            optimizer.step()
            loss_l.append(mse_loss.item())

        # Get activations from last hidden layer
        model.eval()
        with torch.no_grad():
            _, last_hidden = model(X_tensor)
            activations = last_hidden.numpy()

        # PCA and plot
        pca = PCA(n_components=2)
        acts_2d = pca.fit_transform(activations)

        axs_curr_pca.scatter(acts_2d[:,0], acts_2d[:,1], c=labels, cmap='coolwarm')
        axs_curr_pca.axis('equal')
        axs_curr_loss.plot(loss_l)
        axs_curr_loss.set_yscale('log')
plt.tight_layout()
plt.show()