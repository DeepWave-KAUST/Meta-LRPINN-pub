# -----------------------------------------------------------------------------
# Author: Shijun Cheng
# Contact Email: sjcheng.academic@gmail.com
# Date: 2024-12-10
# Description: Meta-training script for METALRPINN. This code performs 
#              meta-training using a MAML-like framework for PDE-based tasks.
#              It loads training data, performs inner and outer optimization 
#              loops, logs progress, periodically saves checkpoints, and 
#              includes validation and testing steps.
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
from utils import PositionalEncod, calculate_grad, normalizer_freq

# Directory for saving checkpoints
dir_checkpoints = './checkpoints/metatrain/'
os.makedirs(dir_checkpoints, exist_ok=True)

def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    
    print(args)

    # Set up TensorBoard writer for logging training/validation metrics
    writer = SummaryWriter(log_dir=f'runs/metatrain/nlayers{args.nlayers}_neurondim{args.neuron_dim}_paramsdim{args.params_dim}_Epoch{args.epoch}')

    device = torch.device('cuda')
    input_dim = 3   # Input dimension (e.g., x, z, sx)
    output_dim = 2  # Output dimension (e.g., real and imaginary parts)

    # Initialize the meta-training model (MAML-like structure)
    maml = METALRPINN_METATRAIN(
        input_dim, output_dim, 
        args.nlayers, args.nlayers_emb, args.rank_dim, args.neuron_dim, args.params_dim
    ).to(device)

    # Count the total number of trainable parameters
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    # -------------------- Load Training Data --------------------
    # Load meta-training data from .mat file. Each column represents a task or a data sample.
    data = sio.loadmat(f'../dataset/metatrain/metatrain.mat')
    x_train = data['x_metatrain']
    sx_train = data['sx_metatrain']
    z_train = data['z_metatrain']
    freq_train = data['freq_metatrain']
    m_train = data['m_metatrain']
    m0_train = data['m0_metatrain']
    u0_real_train = data['u0_real_metatrain']
    u0_imag_train = data['u0_imag_metatrain']

    # Convert numpy arrays to torch tensors and move them to GPU
    x_train = torch.tensor(x_train, dtype=torch.float32, requires_grad=True).to(device)
    z_train = torch.tensor(z_train, dtype=torch.float32, requires_grad=True).to(device)
    sx_train = torch.tensor(sx_train, dtype=torch.float32).to(device)
    freq_train = torch.tensor(freq_train, dtype=torch.float32).to(device)
    m_train = torch.tensor(m_train, dtype=torch.float32).to(device)
    m0_train = torch.tensor(m0_train, dtype=torch.float32).to(device)
    u0_real_train = torch.tensor(u0_real_train, dtype=torch.float32).to(device)
    u0_imag_train = torch.tensor(u0_imag_train, dtype=torch.float32).to(device)

    # -------------------- Load Validation Data --------------------
    # For validation, choose a target frequency and load corresponding data.
    target_freq = 6
    data = sio.loadmat(f'../dataset/metatest/layer_model/{target_freq}Hz_train_data.mat')
    x_val = data['x_train']
    sx_val = data['sx_train']
    z_val = data['z_train']
    m_val = data['m_train']
    m0_val = data['m0_train']
    u0_real_val = data['u0_real_train']
    u0_imag_val = data['u0_imag_train']

    # -------------------- Load Test Data --------------------
    data = sio.loadmat(f'../dataset/metatest/layer_model/{target_freq}Hz_test_data.mat')
    x_test = data['x_test']
    sx_test = data['sx_test']
    z_test = data['z_test']
    du_real_test = data['du_real_test']
    du_imag_test = data['du_imag_test']

    # Convert validation data to torch tensors
    x_val = torch.tensor(x_val, dtype=torch.float32, requires_grad=True).to(device)
    z_val = torch.tensor(z_val, dtype=torch.float32, requires_grad=True).to(device)
    sx_val = torch.tensor(sx_val, dtype=torch.float32).to(device)
    m_val = torch.tensor(m_val, dtype=torch.float32).to(device)
    m0_val = torch.tensor(m0_val, dtype=torch.float32).to(device)
    u0_real_val = torch.tensor(u0_real_val, dtype=torch.float32).to(device)
    u0_imag_val = torch.tensor(u0_imag_val, dtype=torch.float32).to(device)
    freq_val = torch.tensor(target_freq, dtype=torch.float32).unsqueeze(0).to(device)

    # Convert test data to torch tensors
    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True).to(device)
    z_test = torch.tensor(z_test, dtype=torch.float32, requires_grad=True).to(device)
    sx_test = torch.tensor(sx_test, dtype=torch.float32).to(device)
    du_real_test = torch.tensor(du_real_test, dtype=torch.float32).to(device)
    du_imag_test = torch.tensor(du_imag_test, dtype=torch.float32).to(device)

    # Set up the optimizer for meta-training
    meta_optimizer = torch.optim.AdamW(maml.parameters(), lr=args.meta_lr, weight_decay=4e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, step_size=5000, gamma=0.8)

    # Define loss functions
    criterion_pde = PINNLoss()           # PDE residual loss (physics-informed)
    criterion_reg = RegLoss()            # Additional regularization loss
    criterion_reg_ort = OrthogonalityLoss()  # Orthogonality regularization for low-rank bases

    _, data_num = x_train.size()

    # -------------------- Meta-Training Loop --------------------
    for step in range(args.epoch):
        maml.train()

        # Randomly select tasks for support (inner update) and query (outer update)
        rand_task_select = random.sample(range(data_num), args.ntask * 2)
        rand_task_spt = rand_task_select[:args.ntask]   # support tasks
        rand_task_qry = rand_task_select[args.ntask:]   # query tasks

        outer_loss = torch.tensor(0., device=device)

        # Loop over tasks to compute outer_loss for meta-update
        for i in range(args.ntask):
            # Support set data for the i-th task
            x = x_train[:, rand_task_spt[i]].unsqueeze(1)
            z = z_train[:, rand_task_spt[i]].unsqueeze(1)
            sx = sx_train[:, rand_task_spt[i]].unsqueeze(1)
            freq = freq_train[:, rand_task_spt[i]]
            m0 = m0_train[:, rand_task_spt[i]].unsqueeze(1)
            m = m_train[:, rand_task_spt[i]].unsqueeze(1)
            u0_real = u0_real_train[:, rand_task_spt[i]].unsqueeze(1)
            u0_imag = u0_imag_train[:, rand_task_spt[i]].unsqueeze(1)

            # Get a copy of the model parameters for inner updates
            params = OrderedDict(maml.named_parameters())

            # Combine input features
            input = torch.cat([x, z, sx], -1)

            # -------------------- Inner Updates (fast adaptation) --------------------
            for k in range(args.update_step):
                # Forward with functional parameters
                du_real, du_imag = maml.functional_forward(input, normalizer_freq(freq.unsqueeze(1)), params=params)

                # Compute gradients (second derivatives) for PDE constraints
                du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x, z, du_real, du_imag)

                # Compute PDE loss and regularization losses
                loss_pde = criterion_pde(x, z, sx, 2*math.pi*freq, m, m0, u0_real, u0_imag, 
                                         du_real, du_imag, du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)
                loss_reg = criterion_reg(x, z, sx, 2*math.pi*freq, m0, du_real, du_imag)

                # Orthogonality regularization for low-rank bases
                loss_reg_ort = torch.tensor(0, dtype=torch.float32).to(device)
                for ilayers in range(args.nlayers):
                    loss_reg_ort += criterion_reg_ort(params[f'col_basis_{ilayers}'], 
                                                      params[f'row_basis_{ilayers}'], args.rank_dim)

                inner_loss = args.loss_scale * (loss_pde + loss_reg + loss_reg_ort)

                # Compute gradients of inner_loss w.r.t. params
                grads = torch.autograd.grad(inner_loss, params.values(), create_graph=not args.first_order)

                # Update params for this task's inner loop (fast adaptation)
                params = OrderedDict(
                    (name, param - args.update_lr * grad)
                    for ((name, param), grad) in zip(params.items(), grads)
                )

            # -------------------- Outer Update Step --------------------
            # Query set data for the i-th task
            x = x_train[:, rand_task_qry[i]].unsqueeze(1)
            z = z_train[:, rand_task_qry[i]].unsqueeze(1)
            sx = sx_train[:, rand_task_qry[i]].unsqueeze(1)
            freq = freq_train[:, rand_task_qry[i]]
            m0 = m0_train[:, rand_task_qry[i]].unsqueeze(1)
            m = m_train[:, rand_task_qry[i]].unsqueeze(1)
            u0_real = u0_real_train[:, rand_task_qry[i]].unsqueeze(1)
            u0_imag = u0_imag_train[:, rand_task_qry[i]].unsqueeze(1)

            input = torch.cat([x, z, sx], -1)

            # Evaluate on the query set with updated parameters
            du_real, du_imag = maml.functional_forward(input, normalizer_freq(freq.unsqueeze(1)), params=params)

            du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x, z, du_real, du_imag)

            # Compute losses on the query set
            loss_pde = criterion_pde(x, z, sx, 2*math.pi*freq, m, m0, u0_real, u0_imag, 
                                     du_real, du_imag, du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)
            loss_reg = criterion_reg(x, z, sx, 2*math.pi*freq, m0, du_real, du_imag)

            loss_reg_ort = torch.tensor(0, dtype=torch.float32).to(device)
            for ilayers in range(args.nlayers):
                loss_reg_ort += criterion_reg_ort(params[f'col_basis_{ilayers}'], 
                                                  params[f'row_basis_{ilayers}'], args.rank_dim)

            # Accumulate outer loss from all tasks
            outer_loss += args.loss_scale * (loss_pde + loss_reg + loss_reg_ort)

        # Average outer_loss over the number of tasks
        outer_loss = outer_loss / args.ntask

        # Meta-optimizer update on outer_loss
        meta_optimizer.zero_grad()
        outer_loss.backward()
        meta_optimizer.step()
        scheduler.step()

        # Logging to TensorBoard
        writer.add_scalar('Loss/meta_loss', outer_loss.item(), step)
        writer.add_scalar('Loss/loss_pde', loss_pde.item(), step)
        writer.add_scalar('Loss/loss_reg', loss_reg.item(), step)
        writer.add_scalar('Loss/loss_reg_ort', loss_reg_ort.item(), step)
        writer.add_scalar('Loss/inner_loss', inner_loss.item(), step)

        # Print training status every 100 steps
        if (step + 1) % 100 == 0:
            print(f'step: {step + 1} Training inner loss: {inner_loss.item()}')
            print(f'step: {step + 1} Training meta loss: {outer_loss.item()}')
            print(f'step: {step + 1} Training PDE loss: {loss_pde.item()}')
            print(f'step: {step + 1} Training REG loss: {loss_reg.item()}')
            print(f'step: {step + 1} Training ORT loss: {loss_reg_ort.item()}')

        # Save model checkpoint every 100 steps
        if (step + 1) % 100 == 0:
            torch.save(maml.state_dict(), f'{dir_checkpoints}CP_epoch{step + 1}.pth')

        # Perform validation every 5000 steps
        if (step + 1) % 5000 == 0:
            print('---------------------------------------------------------')
            print('------------------- Validation start --------------------')
            print('---------------------------------------------------------')

            # Extract model parameters
            params = OrderedDict(maml.named_parameters())
            start_w = params['start_layer.weight']
            start_b = params['start_layer.bias']
            end_w = params['end_layer.weight']
            end_b = params['end_layer.bias']

            with torch.no_grad():
                alpha_list = maml.meta_forward(freq_val)

            col_basis_list = {}
            row_basis_list = {}
            for ilayers in range(args.nlayers):
                param_name = f'col_basis_{ilayers}'
                col_basis = maml.state_dict()[param_name]
                col_basis_list[param_name] = col_basis

                param_name = f'row_basis_{ilayers}'
                row_basis = maml.state_dict()[param_name]
                row_basis_list[param_name] = row_basis

            # Define meta-test network initialized from meta-trained parameters
            net_val = METALRPINN_METATEST(
                input_dim, output_dim, args.nlayers, args.neuron_dim, 
                start_w, start_b, end_w, end_b, col_basis_list, row_basis_list, alpha_list, 
                is_learn=args.is_learn
            ).to(device)
            val_optimizer = torch.optim.AdamW(net_val.parameters(), lr=args.test_lr, weight_decay=4e-5)

            # Fine-tune on validation data
            for k in range(args.epoch_val):
                val_optimizer.zero_grad()

                coords_val = torch.cat([x_val, z_val, sx_val], -1)
                du_real_pred, du_imag_pred = net_val(coords_val)

                du_real_xx, du_real_zz, du_imag_xx, du_imag_zz = calculate_grad(x_val, z_val, du_real_pred, du_imag_pred)
                loss_pde = criterion_pde(x_val, z_val, sx_val, 2*math.pi*target_freq, m_val, m0_val,
                                         u0_real_val, u0_imag_val, du_real_pred, du_imag_pred,
                                         du_real_xx, du_real_zz, du_imag_xx, du_imag_zz)
                loss_reg = criterion_reg(x_val, z_val, sx_val, 2*math.pi*target_freq, m0_val, du_real_pred, du_imag_pred)

                loss = loss_pde + loss_reg
                loss.backward()
                val_optimizer.step()

                if (k + 1) % 100 == 0:
                    print(f'step: {k + 1} Validation loss: {loss.item()}')
                    print(f'step: {k + 1} Validation PDE loss: {loss_pde.item()}')
                    print(f'step: {k + 1} Validation REG loss: {loss_reg.item()}')

            # Evaluate on test data after validation
            with torch.no_grad():
                input = torch.cat([x_test, z_test, sx_test], -1)
                du_real_pred, du_imag_pred = net_val(input)
                # Mean squared error on test data
                accs_real = (torch.pow((du_real_test - du_real_pred), 2)).mean().item()
                accs_imag = (torch.pow((du_imag_test - du_imag_pred), 2)).mean().item()

            print(f'Test accs real: {accs_real}')
            print(f'Test accs imag: {accs_imag}')

            print('---------------------------------------------------------')
            print('---------------------- Ending Test ----------------------')
            print('---------------------------------------------------------')

    writer.close()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='Number of meta-training iterations', default=50000)
    argparser.add_argument('--epoch_val', type=int, help='Number of epochs for validation/fine-tuning', default=5000)
    argparser.add_argument('--ntask', type=int, help='Number of tasks per meta-iteration', default=5)
    argparser.add_argument('--meta_lr', type=float, help='Meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='Task-level inner update learning rate', default=2e-3)
    argparser.add_argument('--test_lr', type=float, help='Learning rate during meta-testing/validation', default=1e-3)
    argparser.add_argument('--loss_scale', type=float, help='Scaling factor for total losses', default=0.1)
    argparser.add_argument('--update_step', type=int, help='Number of inner update steps', default=1)
    argparser.add_argument('--PosEnc', type=int, help='Positional encoding parameter', default=2)
    argparser.add_argument('--nlayers', type=int, help='Number of hidden layers', default=6)
    argparser.add_argument('--nlayers_emb', type=int, help='Number of FEH embedding layers', default=3)
    argparser.add_argument('--rank_dim', type=int, help='Rank dimension for low-rank decomposition', default=100)
    argparser.add_argument('--neuron_dim', type=int, help='Number of neurons per hidden layer', default=320)
    argparser.add_argument('--params_dim', type=int, help='Dimension of FEH input parameters', default=1)
    argparser.add_argument('--first_order', type=str, help='Use first-order approximation in MAML', default=True)
    argparser.add_argument('--is_learn', type=str, help='Whether to update col/row matrices at meta-test', default=True)

    args = argparser.parse_args()

    main(args)
