import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def adaptive_rank(col_basis_list, row_basis_list, alpha_list, nlayers, keep_scale):
    """
    Prunes or adapts the rank of low-rank decomposed layers based on the keep scale.
    
    For each layer:
    - Finds the top-k largest rank in the alpha vector.
    - Selects corresponding columns/rows in col_basis and row_basis to keep.
    
    Arguments:
    - col_basis_list: Dictionary of column basis for each layer.
    - row_basis_list: Dictionary of row basis for each layer.
    - alpha_list: Dictionary of alpha vectors for each layer.
    - nlayers: Number of layers.
    - keep_scale: keep_scale percent of whole rank

    Returns:
    - col_basis_list, row_basis_list, alpha_list: Updated dictionaries with pruned dimensions.
    """
    for i in range(nlayers):
        param_name = f'alpha_{i}'
        alpha = alpha_list[param_name]
        abs_alpha = torch.abs(alpha)

        # Find the position with the largest absolute value
        k = abs_alpha.size(0) * keep_scale 
        topk_vals, topk_indices = torch.topk(abs_alpha, int(k))
        
        # Keep only the defined scale of whole rank
        alpha_list[param_name] = torch.gather(input=alpha, dim=0, index=topk_indices)

        # Update column basis for this layer
        param_name_col = f'col_basis_{i}'
        col = col_basis_list[param_name_col]
        col_basis_list[param_name_col] = torch.index_select(input=col, dim=1, index=topk_indices)

        # Update row basis for this layer
        param_name_row = f'row_basis_{i}'
        row = row_basis_list[param_name_row]
        # Note: The code seems to use col instead of row by mistake; should be `row` in index_select.
        row_basis_list[param_name_row] = torch.index_select(input=row, dim=0, indextopk_indices)

    return col_basis_list, row_basis_list, alpha_list

class PositionalEncod(nn.Module):
    def __init__(self, PosEnc=2, device='cpu'):
        """
        Positional Encoding module:
        Applies sinusoidal/cosine positional encodings to the input coordinates (x, z, sx).
        
        Arguments:
        - PosEnc: Defines how many frequency bands (powers of 2) to use in the encoding.
        - device: The device on which to create the encoding tensors.
        
        This creates three sets of frequency multipliers (for x, z, sx) and applies 
        sin and cos transformations to enhance feature representation.
        """
        super().__init__()
        self.PEnc = PosEnc
        # Create frequency multiplier arrays for x, z, sx
        self.k_pi_x = (torch.tensor(np.pi)*(2**torch.arange(self.PEnc))).reshape(-1, self.PEnc).to(device); self.k_pi_x = self.k_pi_x.T
        self.k_pi_z = (torch.tensor(np.pi)*(2**torch.arange(self.PEnc))).reshape(-1, self.PEnc).to(device); self.k_pi_z = self.k_pi_z.T
        self.k_pi_sx = (torch.tensor(np.pi)*(2**torch.arange(self.PEnc))).reshape(-1, self.PEnc).to(device); self.k_pi_sx = self.k_pi_sx.T

    def forward(self, input):
        """
        input: Tensor of shape (N, 3), where columns are x, z, sx.
        This method applies sin and cos transformations for each coordinate with different frequencies.
        
        Returns:
        - Tensor with original input features concatenated with positional encodings.
          If input is (N, 3), output will have more dimensions due to encoded features.
        """
        # Apply sin and cos encoding for x
        tmpx = torch.cat([torch.sin(self.k_pi_x*input[:,0]), torch.cos(self.k_pi_x*input[:,0])], axis=0)
        # Apply sin and cos encoding for z
        tmpz = torch.cat([torch.sin(self.k_pi_z*input[:,1]), torch.cos(self.k_pi_z*input[:,1])], axis=0)
        # Apply sin and cos encoding for sx
        tmpsx = torch.cat([torch.sin(self.k_pi_sx*input[:,2]), torch.cos(self.k_pi_sx*input[:,2])], axis=0)

        # Concatenate all encodings: shape (2*PosEnc*3, N), then transpose to (N, ...)
        cat = torch.cat((tmpx, tmpz, tmpsx), axis=0)
        return torch.cat([input, cat.T], -1)

def normalizer_coords(x, dmin, dmax):
    """
    Normalizes coordinates x to the range [-1, 1] given a min and max.
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def normalizer_vel(x, dmin=1.5, dmax=4.6):
    """
    Normalizes velocity values to the range [-1, 1].
    Default min and max velocity values are given.
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def normalizer_freq(x, dmin=2, dmax=20):
    """
    Normalizes frequency values to the range [-1, 1].
    Default min and max frequency values are given.
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def calculate_grad(x, z, du_real, du_imag):
    """
    Calculates second-order derivatives of the predicted real and imaginary wavefields w.r.t. x and z.
    
    Inputs:
    - x, z: spatial coordinates (tensors that require grad)
    - du_real, du_imag: predicted real and imaginary parts of the wavefield
    
    This function computes:
    du_real_xx = d²u_real/dx²
    du_real_zz = d²u_real/dz²
    du_imag_xx = d²u_imag/dx²
    du_imag_zz = d²u_imag/dz²

    Returns:
    - du_real_xx, du_real_zz, du_imag_xx, du_imag_zz: second-order derivatives w.r.t. x and z.
    """
    # First derivatives for real field
    du_real_x = torch.autograd.grad(du_real, x, grad_outputs=torch.ones_like(du_real), create_graph=True)[0]
    du_real_z = torch.autograd.grad(du_real, z, grad_outputs=torch.ones_like(du_real), create_graph=True)[0]
    # Second derivatives for real field
    du_real_xx = torch.autograd.grad(du_real_x, x, grad_outputs=torch.ones_like(du_real_x), create_graph=True)[0]
    du_real_zz = torch.autograd.grad(du_real_z, z, grad_outputs=torch.ones_like(du_real_z), create_graph=True)[0]

    # First derivatives for imaginary field
    du_imag_x = torch.autograd.grad(du_imag, x, grad_outputs=torch.ones_like(du_imag), create_graph=True)[0]
    du_imag_z = torch.autograd.grad(du_imag, z, grad_outputs=torch.ones_like(du_imag), create_graph=True)[0]
    # Second derivatives for imaginary field
    du_imag_xx = torch.autograd.grad(du_imag_x, x, grad_outputs=torch.ones_like(du_imag_x), create_graph=True)[0]
    du_imag_zz = torch.autograd.grad(du_imag_z, z, grad_outputs=torch.ones_like(du_imag_z), create_graph=True)[0]

    return du_real_xx, du_real_zz, du_imag_xx, du_imag_zz
