import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, fixed_output=False, linear_net=False, G=1, bias=False):
        super(DNN, self).__init__()
        self.num_layers = num_layers
        self.bias = bias

        # Define layers
        if num_layers == 0:
            self.output_layer = nn.Linear(input_size, output_size, bias=bias)
        else:
            self.input_layer = nn.Linear(input_size, hidden_size, bias=bias)
            self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=bias) for _ in range(num_layers - 1)])
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
        if self.num_layers == 0:
            nn.init.xavier_normal_(self.output_layer.weight, gain=G)
        else:
            nn.init.xavier_normal_(self.input_layer.weight, gain=G)
            
            for layer in self.hidden_layers:
                nn.init.xavier_normal_(layer.weight, gain=G)
            if fixed_output:
                nn.init.normal_(self.output_layer.weight)
            else:
                nn.init.xavier_normal_(self.output_layer.weight, gain=G)
            
    def forward(self, x):
        if self.num_layers == 0:
            return self.output_layer(x), [x]
        hidden_states = []

        # Input layer
        x = self.activation(self.input_layer(x))
        hidden_states.append(x)

        # Hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            hidden_states.append(x)

        # Output layer
        out = self.output_layer(x)

        return out, hidden_states
