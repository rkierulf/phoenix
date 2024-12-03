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

import torch
import torch.optim as optim

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

#from datagenerator import DataGenerator
from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
#from visualization_inte import *
import matplotlib.pyplot as plt

#torch.set_num_threads(4) #CHANGE THIS!

def make_mask(X):
    triu = np.triu(X)
    tril = np.tril(X)
    triuT = triu.T
    trilT = tril.T
    masku = abs(triu) > abs(trilT)
    maskl = abs(tril) > abs(triuT)
    main_mask = ~(masku | maskl)
    X[main_mask] = 0

#From https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-024-03264-0/MediaObjects/13059_2024_3264_MOESM2_ESM.pdf

def extract_dynamics_matrix(output_root_dir):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sums_path = os.path.join(current_dir, output_root_dir, 'best_val_model_sums.pt')
    prods_path = os.path.join(current_dir, output_root_dir, 'best_val_model_prods.pt')
    alpha_comb_path = os.path.join(current_dir, output_root_dir, 'best_val_model_alpha_comb.pt')
    gene_mult_path = os.path.join(current_dir, output_root_dir, 'best_val_model_gene_multipliers.pt')
    sums_model = torch.load(sums_path)
    prods_model = torch.load(prods_path)
    alpha_comb = torch.load(alpha_comb_path)
    gene_mult = torch.load(gene_mult_path)
    extract_dynamics(output_root_dir, sums_model, prods_model, alpha_comb, gene_mult)

def extract_dynamics(output_root_dir, sums_model, prods_model, alpha_comb, gene_mult):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    Wo_sums = np.transpose(sums_model.linear_out.weight.detach().numpy())
    Bo_sums = np.transpose(sums_model.linear_out.bias.detach().numpy())
    Wo_prods = np.transpose(prods_model.linear_out.weight.detach().numpy())
    Bo_prods = np.transpose(prods_model.linear_out.bias.detach().numpy())
    alpha_comb = np.transpose(alpha_comb.linear_out.weight.detach().numpy())
    gene_mult = np.transpose(torch.relu(gene_mult.detach()).numpy())

    num_features = alpha_comb.shape[0]
    effects_mat = np.matmul(Wo_sums,alpha_comb[0:num_features//2]) + np.matmul(Wo_prods,alpha_comb[num_features//2:num_features])
    num_cols = effects_mat.shape[1]
    effects_mat = effects_mat * np.transpose(gene_mult)
    effects_mat_path = os.path.join(current_dir, output_root_dir, 'effects_mat.csv')
    np.savetxt(effects_mat_path, effects_mat, delimiter=",")

    dynamics_mat = np.zeros((num_cols, num_cols))
    effects_sums = (np.abs(effects_mat)).sum(axis=1)
    for i in range(num_cols):
        for j in range(num_cols):
            dynamics_mat[i,j] = effects_mat[i,j] / (effects_sums[j] - abs(effects_mat[i,j]))

    dynamics_mat_path = os.path.join(current_dir, output_root_dir, 'dynamics_mat.csv')
    np.savetxt(dynamics_mat_path, dynamics_mat, delimiter=",")
