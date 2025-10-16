# -----------------------------------------------------------------------------
# Author: Shijun Cheng
# Contact Email: sjcheng.academic@gmail.com
# Date: 2024-12-10
# Description: Model definitions for the Meta-LRPINN framework.
#              This file includes the meta-training and meta-testing network 
#              architectures, as well as a baseline VanillaPINN model.
#              The METALRPINN_METATRAIN class generates layers with low-rank 
#              decomposed weights and uses a frequency embedding hypernetwork (FEH)
#              to produce alpha parameters that modulate the hidden layers.
#              The METALRPINN_METATEST class is used for meta-testing by loading 
#              parameters learned during meta-training. The VanillaPINN is provided 
#              as a baseline model without meta-learning or low-rank decomposition.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.init as init
from   torch.nn import functional as F

# The network for meta-training phase
class METALRPINN_METATRAIN(nn.Module):
    def __init__(self, input_dim, output_dim, nlayers, nlayers_emb, rank_dim, neuron_dim, params_dim):
        """
        Arguments:
        - input_dim: Dimension of input features
        - output_dim: Dimension of output features (e.g., 2 for predicting two quantities)
        - nlayers: Number of hidden layers (low-rank decomposition layers)
        - nlayers_emb: Number of layers in the Frequency Embedding Hypernetwork (FEH)
        - rank_dim: Rank dimension used for low-rank decomposition of each layer's weight
        - neuron_dim: Number of neurons in each hidden layer
        - params_dim: Dimension of the parameters fed into the FEH (meta-parameters)
        
        This model uses a Frequency Embedding Hypernetwork (FEH) to generate alpha parameters
        that modulate the hidden layers' weights represented via low-rank decomposition.
        """
        super(METALRPINN_METATRAIN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.nlayers_emb = nlayers_emb
        self.rank_dim = rank_dim
        self.neuron_dim = neuron_dim
        self.params_dim = params_dim

        # Define input and output linear layers
        self.start_layer = nn.Linear(input_dim, neuron_dim)
        self.end_layer = nn.Linear(neuron_dim, output_dim)

        # For each hidden layer, create low-rank decomposition parameters: col_basis and row_basis
        # and also a meta_alpha layer that maps from the FEH embedding to alpha parameters.
        for i in range(nlayers):
            # col_basis (neuron_dim, rank_dim)
            param_name = f'col_basis_{i}'
            param_value = nn.Parameter(torch.rand(self.neuron_dim, self.rank_dim))
            init.xavier_normal_(param_value, gain=1)
            setattr(self, param_name, param_value)

            # row_basis (rank_dim, neuron_dim)
            param_name = f'row_basis_{i}'
            param_value = nn.Parameter(torch.rand(self.rank_dim, self.neuron_dim))
            init.xavier_normal_(param_value, gain=1)
            setattr(self, param_name, param_value)

            # meta_alpha layer takes the FEH output and produces the alpha vector for this layer
            layer_name = f'meta_alpha_{i}'
            layer = nn.Linear(self.neuron_dim//4, self.rank_dim)
            setattr(self, layer_name, layer)

        # Build the FEH layers: they map params_dim -> neuron_dim//4 through multiple sinusoidal transformations
        for i in range(nlayers_emb):
            if i == 0:
                self.meta_layer_0 = nn.Linear(self.params_dim, self.neuron_dim//4)
            else:
                layer_name = f'meta_layer_{i}'
                layer = nn.Linear(self.neuron_dim//4, self.neuron_dim//4)
                setattr(self, layer_name, layer)

        # Non-linear activation function for FEH
        self.gelu = nn.GELU()

    # Frequency embedding hypernetwork (FEH)
    def meta_forward(self, input_params):
        """
        The FEH transforms the input parameters into a representation used to generate alpha vectors.
        It applies multiple layers with sine activations.
        
        Returns:
        - alpha_list: A dictionary where each key is 'alpha_i' corresponding to each layer's alpha vector.
        """
        # Pass through nlayers_emb FEH layers
        for i in range(self.nlayers_emb):
            layer = getattr(self, f'meta_layer_{i}')
            input_params = layer(input_params)
            input_params = torch.sin(input_params)

        # Generate alpha vector for each layer
        alpha_list = {}
        for i in range(self.nlayers):
            param_name = f'alpha_{i}'
            layer = getattr(self, f'meta_alpha_{i}')
            output = self.gelu(layer(input_params))
            alpha_list[param_name] = output.squeeze()

        return alpha_list

    # The hidden layer of LRPINN, where LRPINN includes frequency-aware weights
    def hiddenlayer_forward(self, emb_out, alpha_list):
        """
        Forward pass through the hidden layers using low-rank decomposition and alpha parameters.
        
        Arguments:
        - emb_out: The embedding (features) from the previous layer
        - alpha_list: Dictionary of alpha vectors for each layer

        Returns:
        - emb_out: The output after passing through all hidden layers
        - col_basis_list, row_basis_list: Dictionaries containing the low-rank basis parameters for each layer
        """
        col_basis_list = {}
        row_basis_list = {}
        for i in range(self.nlayers):
            # Retrieve the col_basis and row_basis for this layer
            param_name = f'col_basis_{i}'
            col_basis = getattr(self, param_name)
            col_basis_list[param_name] = col_basis

            param_name = f'row_basis_{i}'
            row_basis = getattr(self, param_name)
            row_basis_list[param_name] = row_basis

            # Reconstruct the full weight matrix W = col_basis * diag(alpha) * row_basis
            alpha = alpha_list[f'alpha_{i}']
            weight = torch.matmul(torch.matmul(col_basis, torch.diag(alpha)), row_basis)

            # Apply the layer transformation: emb_out * W, then sine activation
            emb_out = torch.matmul(emb_out, weight)
            emb_out = torch.sin(emb_out)

        return emb_out, col_basis_list, row_basis_list

    def forward(self, inputs, input_params):
        """
        Main forward pass:
        1. Use FEH (meta_forward) to get alpha parameters.
        2. Process input through the input layer.
        3. Pass through hidden layers using low-rank decomposition and alpha.
        4. Output layer to produce final predictions.
        
        Returns:
        - emb_out[:, 0:1], emb_out[:, 1:2]: The two output channels
        - col_basis_list, row_basis_list: Low-rank decomposition parameters (useful if needed externally)
        """
        ##### FEH output #####
        alpha_list = self.meta_forward(input_params)

        # input layer
        emb_out = self.start_layer(inputs)
        emb_out = torch.sin(emb_out)

        # hidden layer
        emb_out, col_basis_list, row_basis_list = self.hiddenlayer_forward(emb_out, alpha_list)

        # final layer
        emb_out = self.end_layer(emb_out)
        emb_out = emb_out
        
        return emb_out[:, 0:1], emb_out[:, 1:2], col_basis_list, row_basis_list

    # A functional forward function
    def functional_forward(self, input, input_params, params):
        """
        A functional forward pass (useful for meta-learning approaches like MAML).
        Instead of using the internal parameters, it uses an externally provided 'params' dict.
        
        Arguments:
        - input: Input features
        - input_params: Input parameters for FEH
        - params: A dictionary of parameters (weights & biases) to be used instead of the internal state_dict
        
        Returns:
        - emb_out[:, 0:1], emb_out[:, 1:2]: The two output channels
        """

        # FEH processing with provided params
        for i in range(self.nlayers_emb): 
            input_params = Functional_linear(input_params, params[f'meta_layer_{i}.weight'], 
                        params[f'meta_layer_{i}.bias'])
            input_params = torch.sin(input_params)

        # Input layer with provided params
        emb_out = Functional_linear(input, params['start_layer.weight'], params['start_layer.bias'])
        emb_out = torch.sin(emb_out)

        # Hidden layers
        for i in range(self.nlayers):
            # Compute alpha for this layer
            alpha = Functional_linear(input_params, params[f'meta_alpha_{i}.weight'], 
                        params[f'meta_alpha_{i}.bias'])
            alpha = (self.gelu(alpha)).squeeze()

            # Reconstruct weight matrix
            weight = torch.matmul(torch.matmul(params[f'col_basis_{i}'], 
                        torch.diag(alpha)), params[f'row_basis_{i}'])

            # Apply layer
            emb_out = torch.sin(torch.matmul(emb_out, weight))

        # Output layer
        emb_out = F.linear(emb_out, params['end_layer.weight'], params['end_layer.bias'])
        return emb_out[:, 0:1], emb_out[:, 1:2]

def Functional_linear(inp, weight, bias):
    """
    A functional linear layer for use in functional_forward, followed by a sine activation.
    Equivalent to: F.linear(inp, weight, bias) -> sin.
    """
    inp = F.linear(inp, weight, bias)
    inp = torch.sin(inp)
    return inp

# The network for meta-testing phase
class METALRPINN_METATEST(nn.Module):
    def __init__(self, input_dim, output_dim, nlayers, neuron_dim, start_w, start_b, 
                end_w, end_b, col_basis_list, row_basis_list, alpha_list, is_learn=False):
        """
        Arguments:
        - input_dim, output_dim, nlayers, neuron_dim: same as meta-train stage
        - start_w, start_b, end_w, end_b: Learned parameters for input and output layers from meta-train
        - col_basis_list, row_basis_list, alpha_list: Learned low-rank parameters and alpha from meta-train
        - is_learn: Whether to allow these parameters to be trainable during meta-test (e.g., fine-tuning)
        
        In meta-testing, we load the learned parameters from meta-training and use them to make predictions
        on a new task quickly.
        """
        super(METALRPINN_METATEST, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.neuron_dim = neuron_dim

        # Set up input and output layers with trained parameters
        self.start_layer = nn.Linear(input_dim, neuron_dim)
        self.end_layer = nn.Linear(neuron_dim, output_dim)
      
        self.start_layer.weight = nn.Parameter(start_w)
        self.start_layer.bias = nn.Parameter(start_b)
        self.end_layer.weight = nn.Parameter(end_w)
        self.end_layer.bias = nn.Parameter(end_b)

        # Load the trained col_basis, row_basis, and alpha for each hidden layer  
        for i in range(nlayers):
            param_name = f'col_basis_{i}'
            param_value = nn.Parameter(col_basis_list[param_name], requires_grad=is_learn)
            setattr(self, param_name, param_value)

            param_name = f'row_basis_{i}'
            param_value = nn.Parameter(row_basis_list[param_name], requires_grad=is_learn)
            setattr(self, param_name, param_value)
    
            param_name = f'alpha_{i}'
            param_value = nn.Parameter(alpha_list[param_name])
            setattr(self, param_name, param_value)

    def forward(self, inputs):
        """
        Forward pass for meta-test:
        Uses the parameters learned during meta-training and potentially fine-tuned during meta-test.
        """

        # Input layer
        emb_out = self.start_layer(inputs)
        emb_out = torch.sin(emb_out)

        # Hidden layers: reconstruct weights using low-rank decomposition and alpha
        for i in range(self.nlayers):
            param_name = f'col_basis_{i}'
            col_basis = getattr(self, param_name)

            param_name = f'row_basis_{i}'
            row_basis = getattr(self, param_name)

            param_name = f'alpha_{i}'
            alpha = getattr(self, param_name)

            # Full weight matrix
            weight = torch.matmul(torch.matmul(col_basis, torch.diag(alpha)), row_basis)

            # Apply layer
            emb_out = torch.matmul(emb_out, weight)
            emb_out = torch.sin(emb_out)

        # Output layer
        emb_out = self.end_layer(emb_out)

        return emb_out[:, 0:1], emb_out[:, 1:2]

# A vanilla PINN model for baseline comparison (no meta-learning, no low-rank decomposition)
class VanillaPINN(nn.Module):
    def __init__(self, input_dim, output_dim, nlayers, neuron_dim):
        """
        A standard PINN:
        - Fully-connected layers
        - Sine activations
        - No meta-learning or low-rank decomposition
        """
        super(VanillaPINN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.neuron_dim = neuron_dim

        self.start_layer = nn.Linear(input_dim, neuron_dim)
        self.end_layer = nn.Linear(neuron_dim, output_dim)

        # Create hidden layers
        for i in range(nlayers):
            layer_name = f'hidden_layer{i}'
            layer = nn.Linear(neuron_dim, neuron_dim)
            setattr(self, layer_name, layer)

    def forward(self, x):
        # Input layer
        emb_out = self.start_layer(x)
        emb_out = torch.sin(emb_out)

        # Hidden layers
        for i in range(self.nlayers):
            layer_name = f'hidden_layer{i}'
            layer = getattr(self, layer_name)
            emb_out = layer(emb_out)
            emb_out = torch.sin(emb_out)

        # Output layer
        emb_out = self.end_layer(emb_out)
        
        return emb_out[:, 0:1], emb_out[:, 1:2]