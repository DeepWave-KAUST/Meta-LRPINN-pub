# -----------------------------------------------------------------------------
# Author: Shijun Cheng
# Date: 2024-12-10
# Description:
# This script performs meta-testing (or validation/testing phase) for the 
# meta-trained METALRPINN model at a given frequency. It:
# 1. Loads a pre-trained model checkpoint (from meta-training).
# 2. Loads validation and test data corresponding to a specific target frequency.
# 3. Optionally performs adaptive rank adjustments.
# 4. Fine-tunes (if is_learn=True) the model on the validation set.
# 5. Periodically evaluates the model on the test set and saves predictions, logs, and checkpoints.
# -----------------------------------------------------------------------------

import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import argparse
import scipy.io as sio
from torch.utils.tensorboard import SummaryWriter
from models import METALRPINN_METATRAIN, METALRPINN_METATEST
from copy import deepcopy
from random import shuffle
import math
from collections import OrderedDict
import random
from losses import PINNLoss, RegLoss, OrthogonalityLoss
from utils import PositionalEncod, calculate_grad, normalizer_freq, adaptive_rank

# Paths to meta-trained checkpoint, meta-testing checkpoint, output directories, etc.
dir_meta = './checkpoints/metatrain/meta_trained.pth'
dir_checkpoints = './checkpoints/metatest/freq'
dir_output = './output/freq'

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    
    print(args)
    device = torch.device('cuda')

    # Define input/output dimensions for the model
    input_dim = 3  # (x, z, sx)
    output_dim = 2 # (real part, imaginary part of the wavefield)

    # Load meta-trained model
    meta = METALRPINN_METATRAIN(input_dim, output_dim, args.nlayers, args.nlayers_emb, 
                                args.rank_dim, args.neuron_dim, args.params_dim).to(device)
    meta.load_state_dict(torch.load(dir_meta, map_location=device))

    # Count number of trainable parameters
    tmp = filter(lambda x: x.requires_grad, meta.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    # ------------------- Load Validation Data for a specific frequency -------------------
    target_freq = 6
    print(f'This is testing for {target_freq} Hz wavefield')
    os.makedirs(f'{dir_output}{target_freq}', exist_ok=True)
    os.makedirs(f'{dir_checkpoints}{target_freq}', exist_ok=True)

    # Load validation (train_data) from .mat file
    # These data represent conditions for fine-tuning the model
    data = sio.loadmat(f'../../dataset/layer_model/{target_freq}Hz_train_data.mat')
    x_val = data['x_train']
    sx_val = data['sx_train']
    z_val = data['z_train']
    m_val = data['m_train']
    m0_val = data['m0_train']
    u0_real_val = data['u0_real_train']
    u0_imag_val = data['u0_imag_train']

    # ------------------- Load Test Data -------------------
    # These data are used for evaluating the model's performance after fine-tuning
    data = sio.loadmat(f'../../dataset/layer_model/{target_freq}Hz_test_data.mat')
    x_test = data['x_test']
    sx_test = data['sx_test']
    z_test = data['z_test']
    du_real_test = data['du_real_test']
    du_imag_test = data['du_imag_test']

    # Convert numpy arrays to torch tensors and move them to the GPU
    x_val = torch.tensor(x_val, dtype=torch.float32, requires_grad=True).to(device)
    z_val = torch.tensor(z_val, dtype=torch.float32, requires_grad=True).to(device)
    sx_val = torch.tensor(sx_val, dtype=torch.float32).to(device)
    m_val = torch.tensor(m_val, dtype=torch.float32).to(device)
    m0_val = torch.tensor(m0_val, dtype=torch.float32).to(device)
    u0_real_val = torch.tensor(u0_real_val, dtype=torch.float32).to(device)
    u0_imag_val = torch.tensor(u0_imag_val, dtype=torch.float32).to(device)
    freq_val = torch.tensor(target_freq, dtype=torch.float32).unsqueeze(0).to(device)

    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True).to(device)
    z_test = torch.tensor(z_test, dtype=torch.float32, requires_grad=True).to(device)
    sx_test = torch.tensor(sx_test, dtype=torch.float32).to(device)
    du_real_test = torch.tensor(du_real_test, dtype=torch.float32).to(device)
    du_imag_test = torch.tensor(du_imag_test, dtype=torch.float32).to(device)

    # Concatenate coordinates for validation and test sets
    coords_val = torch.cat((x_val, z_val, sx_val), dim=-1)
    coords_test = torch.cat((x_test, z_test, sx_test), dim=-1)

    # Extract parameters from meta-trained model
    params = OrderedDict(meta.named_parameters())
    start_w = params['start_layer.weight']
    start_b = params['start_layer.bias']
    end_w = params['end_layer.weight']
    end_b = params['end_layer.bias']

    # Compute alpha_list from the FEH given the target frequency
    with torch.no_grad():
        alpha_list = meta.meta_forward(normalizer_freq(freq_val))

    # Extract low-rank bases from the meta-trained model
    col_basis_list = {}
    row_basis_list = {}
    for i in range(args.nlayers):
        param_name = f'col_basis_{i}'
        col_basis = meta.state_dict()[param_name]
        col_basis_list[param_name] = col_basis

        param_name = f'row_basis_{i}'
        row_basis = meta.state_dict()[param_name]
        row_basis_list[param_name] = row_basis

    alpha = alpha_list['alpha_1']
    print(f'Before adaprank len of rank: {alpha.shape}')
    
    # Optional: Adaptive rank reduction if is_adaprank is True
    if args.is_adaprank:
        print(f'Keep rank scale {args.keep_scale}')
        col_basis_list, row_basis_list, alpha_list = adaptive_rank(col_basis_list, 
                                                                  row_basis_list, 
                                                                  alpha_list, 
                                                                  args.nlayers,
                                                                  args.keep_scale)
        rank_list = []
        for ilayers in range(args.nlayers):
            alpha = alpha_list[f'alpha_{ilayers}']
            rank_list.append(alpha.shape)
        print(f'After adaprank len of rank: {rank_list}')

    # Create METALRPINN_METATEST model for fine-tuning on validation data
    # This model uses the learned parameters from meta-training and can further fine-tune them
    net_val = METALRPINN_METATEST(input_dim, output_dim, args.nlayers, args.neuron_dim, 
                                  start_w, start_b, end_w, end_b, 
                                  col_basis_list, row_basis_list, alpha_list, 
                                  is_learn=args.is_learn).to(device)

    # Set up optimizer and learning rate scheduler for validation fine-tuning
    val_optimizer = torch.optim.AdamW(net_val.parameters(), lr=args.test_lr, weight_decay=4e-5)
    milestones = [2000, 4000, 8000]
    gamma = 0.5
    scheduler = torch.optim.lr_scheduler.MultiStepLR(val_optimizer, milestones=milestones, gamma=gamma)
    print('milestones:', milestones)
    print('gamma:', gamma)

    # Define loss functions
    criterion_pde = PINNLoss()
    criterion_reg = RegLoss()
    criterion_reg_ort = OrthogonalityLoss()

    # Set up TensorBoard writer for logging fine-tuning and testing results
    writer = SummaryWriter(log_dir=f'runs/metatest/Freq{target_freq}_Epoch{args.epoch_val}')

    # Fine-tuning loop over validation data
    for k in range(args.epoch_val):
        net_val.train()
        val_optimizer.zero_grad()

        # Forward pass on validation data
        du_real_pred, du_imag_pred = net_val(coords_val)

        # Compute second derivatives for PDE residual
        du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x_val, z_val, du_real_pred, du_imag_pred)

        # PDE loss and regularization loss
        loss_pde = criterion_pde(x_val, z_val, sx_val, 2*math.pi*target_freq, m_val, m0_val,  
                                 u0_real_val, u0_imag_val, du_real_pred, du_imag_pred, 
                                 du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)

        loss_reg = criterion_reg(x_val, z_val, sx_val, 2*math.pi*target_freq, m0_val, du_real_pred, du_imag_pred)

        # Combine losses with scaling factor
        loss = args.loss_scale * (loss_pde + loss_reg)

        # If is_learn=True, also include orthogonality regularization on updated bases
        if args.is_learn:
            loss_reg_ort = torch.tensor(0, dtype=torch.float32).to(device)
            for i in range(args.nlayers):
                loss_reg_ort += criterion_reg_ort(net_val.state_dict()[f'col_basis_{i}'], 
                                                  net_val.state_dict()[f'row_basis_{i}'], 
                                                  args.rank_dim)
            loss += args.loss_scale * loss_reg_ort

        # Backpropagation and parameter update
        loss.backward()
        val_optimizer.step()

        # Evaluate on test data
        with torch.no_grad():
            net_val.eval()
            du_real_pred, du_imag_pred = net_val(coords_test)
            accs_real = (torch.pow((du_real_test - du_real_pred),2)).mean().item()
            accs_imag = (torch.pow((du_imag_test - du_imag_pred),2)).mean().item()

        # Log PDE loss and Accuracy to TensorBoard
        writer.add_scalar('loss_pde', loss_pde.item(), k)
        writer.add_scalar('accs_real', accs_real, k)
        writer.add_scalar('accs_imag', accs_imag, k)

        # Every 100 steps, evaluate on test data and save predictions
        if (k + 1) % 100 == 0:
            print(f'step: {k + 1} Training loss: {loss.item()}')
            print(f'step: {k + 1} Training PDE loss: {loss_pde.item()}')
            print(f'step: {k + 1} Training REG loss: {loss_reg.item()}')
            if args.is_learn:
                print(f'step: {k + 1} Training REG ORT loss: {loss_reg_ort.item()}')

            # Save predictions and accuracy metrics to .mat file
            sio.savemat(f'{dir_output}{target_freq}/pred{k+1}.mat', 
                        {'du_real_pred': du_real_pred.cpu().numpy(), 
                         'accs_real': accs_real, 
                         'du_imag_pred': du_imag_pred.cpu().numpy(), 
                         'accs_imag': accs_imag})

            print(f'Test accs real: {accs_real}')
            print(f'Test accs imag: {accs_imag}')

            # Save current model checkpoint
            torch.save(net_val.state_dict(), f'{dir_checkpoints}{target_freq}/CP_epoch{k + 1}.pth')

        # Update the learning rate
        scheduler.step()

    print('---------------------------------------------------------')
    print('---------------------- Ending Test ----------------------')
    print('---------------------------------------------------------')

    writer.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch_val', type=int, help='Number of fine-tuning epochs', default=10000)
    argparser.add_argument('--test_lr', type=float, help='Validation/fine-tuning learning rate', default=1e-3)
    argparser.add_argument('--loss_scale', type=float, help='Scaling factor for total losses', default=0.1)
    argparser.add_argument('--regloss_scale', type=float, help='Regularization loss scale', default=1.0)
    argparser.add_argument('--regortloss_scale', type=float, help='Orthogonality loss scale', default=1.0)
    argparser.add_argument('--PosEnc', type=int, help='Positional Encoding parameter', default=4)
    argparser.add_argument('--nlayers', type=int, help='Number of hidden layers', default=6)
    argparser.add_argument('--nlayers_emb', type=int, help='Number of FEH layers', default=3)
    argparser.add_argument('--rank_dim', type=int, help='Rank dimension for low-rank decomposition', default=100)
    argparser.add_argument('--neuron_dim', type=int, help='Number of neurons per layer', default=320)
    argparser.add_argument('--hidden_dim', type=int, help='(Unused here) hidden_dim', default=100)
    argparser.add_argument('--params_dim', type=int, help='Dimension of parameters for FEH', default=1)
    argparser.add_argument('--keep_scale', type=float, help='keep_scale percent of whole rank', default=0.5)
    argparser.add_argument('--is_learn', type=str, help='Whether to update col/row matrices during fine-tuning', default=True)
    argparser.add_argument('--is_adaprank', type=str, help='Whether to apply adaptive rank pruning', default=False)

    args = argparser.parse_args()
    main(args)
