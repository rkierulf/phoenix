# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np
from tqdm import tqdm
from math import ceil
from time import perf_counter, process_time
from extract_model_matrix_PHOENIX import extract_dynamics_matrix
from get_percentiles import get_percentiles
from validate_model import validate_model
import matplotlib.pyplot as plt
from extract_model_matrix_PHOENIX import extract_dynamics
from get_percentiles import get_percentiles
from validate_model import validate_model

import torch
import torch.optim as optim

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

#from datagenerator import DataGenerator
from datahandler import DataHandler
from odenet2 import ODENet2
from read_config import read_arguments_from_file
#from solve_eq import solve_eq
from visualization import *

def read_data_csv(data_file_loc):
    data = np.genfromtxt(data_file_loc, delimiter=',')
    data_torch = torch.from_numpy(data)
    return data_torch.float()

def train_epoch(odenet, data, time, method):
    time_indices = np.arange(0, len(time)-2)
    np.random.shuffle(time_indices)
    total_loss = 0
    mse_loss = torch.nn.MSELoss()
    for i in range(len(time_indices)):
        index = time_indices[i]
        data_point = data[:,index].unsqueeze(0)
        target = data[:,index+1]
        num_time_points = 10
        time_range = torch.from_numpy(np.linspace(time[index], time[index+1], num_time_points)).float()
        opt.zero_grad()
        predictions = odeint(odenet, data_point, time_range, method=method)
        loss = mse_loss(predictions, target)
        loss.backward()
        opt.step()
        total_loss += loss
    return total_loss.item()

# Main function
if __name__ == "__main__":
    print('Setting recursion limit to 3000')
    sys.setrecursionlimit(3000)
    settings_file_loc ='config_dynamical_system.cfg'
    print('Loading settings from file {}'.format(settings_file_loc))
    settings = read_arguments_from_file(settings_file_loc)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    odenet = ODENet2(device, 5, False, settings['neurons_per_layer'])
    odenet.float()
    param_count = sum(p.numel() for p in odenet.parameters() if p.requires_grad)
    param_ratio = round(param_count/ (5)**2, 3)
    print("Using a NN with {} neurons per layer, with {} trainable parameters, i.e. parametrization ratio = {}".format(settings['neurons_per_layer'], param_count, param_ratio))
    print('Using optimizer: {}'.format(settings['optimizer']))

    # Select optimizer
    print('Using optimizer: {}'.format(settings['optimizer']))
    if settings['optimizer'] == 'rmsprop':
        opt = optim.RMSprop(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'sgd':
        opt = optim.SGD(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'adagrad':
        opt = optim.Adagrad(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    else:
        opt = optim.Adam([
                {'params': odenet.net_sums.linear_out.weight}, 
                {'params': odenet.net_sums.linear_out.bias},
                {'params': odenet.net_prods.linear_out.weight},
                {'params': odenet.net_prods.linear_out.bias},
                {'params': odenet.net_alpha_combine.linear_out.weight},
                {'params': odenet.gene_multipliers,'lr': 5*settings['init_lr']}
                
            ],  lr=settings['init_lr'], weight_decay=settings['weight_decay'])


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
    factor=0.9, patience=6, threshold=1e-07, 
    threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-09, verbose=True)
    data_file_loc = "../../dynamical_system_data/variance_015_data.csv"
    train_data = read_data_csv(data_file_loc)
    train_time = np.arange(0, 11)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "dynamical_system_output/variance_015")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    loss_values = []
    for i in range(settings['epochs']):
        print(f"Epoch: {i+1}")
        loss = train_epoch(odenet, train_data, train_time, settings['method'])
        print("MSE Loss: " + str(loss))
        loss_values.append(loss)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss_values, label='Training Loss', color='b')
    plot_path = os.path.join(output_path, 'MSE_loss.png')
    plt.savefig(plot_path)


    print("Extracting dynamics matrix")
    extract_dynamics(output_path, odenet.net_prods, odenet.net_sums, odenet.net_alpha_combine, odenet.gene_multipliers)

    print("Obtaining percentiles from dynamics matrix")
    get_percentiles(output_path)

    print("Comparing with validation network")
    validation_network_path = '../../dynamical_system_data/validation_network.csv'
    genes_path = '../../dynamical_system_data/names.csv'
    validate_model(output_path, genes_path, validation_network_path)

