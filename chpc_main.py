import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler

import argparse

import pyro

from pyro.infer import NUTS, MCMC

import os

from TurbulenceNetwork import TurbulenceNetworkBayesian
from TurbulenceClosureDataGenerator import TurbulenceClosureDataGenerator

from Model_utils import *
from Data_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(
        prog='chpc_main.py',
        description='Train a Bayesian neural network for turbulence closure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--chkpt_dir', '-d', type=str, default='Model_Checkpoints')
    parser.add_argument('--num_chains', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--n_data', type=int, default=120_000)
    parser.add_argument('--tree_depth', type=int, default=12)
    parser.add_argument('--param_file', type=str, default='params.json')
    parser.add_argument('--verbose', '-v', type=int, default=1)

    return parser.parse_args()

def train(parser, params):
    if parser.verbose:
        print(f"Creating {parser.n_data} datapoints")
    etas_train, gs_train = get_data(parser.n_data)

    if parser.verbose:
        print("Scaling Data")
    x_scaler = StandardScaler().fit(etas_train)
    y_scaler = StandardScaler().fit(gs_train)

    if parser.verbose:
        print(f"Using device: {device}")
    x = torch.tensor(x_scaler.transform(etas_train), dtype=torch.float32, device=device)
    y = torch.tensor(y_scaler.transform(gs_train), dtype=torch.float32, device=device)

    model = TurbulenceNetworkBayesian(**params).to(device)

    nuts_kernel = NUTS(model, max_tree_depth=parser.tree_depth)

    mcmc = MCMC(nuts_kernel, num_samples=parser.num_samples, warmup_steps=parser.warmup_steps, num_chains=parser.num_chains, mp_context="spawn")

    if parser.verbose:
        print("---> Beginning Training")

    mcmc.run(x, y)

    save_name = f"MCMC_model_{parser.num_samples}_{parser.warmup_steps}_{parser.tree_depth}_{parser.num_chains}"

    if parser.verbose:
        print("---> Training Complete")
        print(f"---> Saving MCMC object to {parser.chkpt_dir} as {save_name}")

    save_MCMC_model(mcmc, parser.chkpt_dir, save_name)

    if parser.verbose:
        print("---> Successfully Saved MCMC Object")

    

def get_data(n_points):
    SSG_gen = TurbulenceClosureDataGenerator('SSG')

    # Defining the ranges for the different scales
    log = (-.5, 0)

    # Defining whether to exclude certain areas of the data
    exclude_area = False
    include_area = False
    drop_eta_1 = False
    drop_eta_2 = False
    add_noise = False

    # Defining area to exclude datapoints
    eta_1_range = (10**np.array([-.3, 0.0]))**2
    eta_2_range = (10**np.array([-.3, 0.0]))**2

    etas_train, gs_train = generate_log_data(SSG_gen, log, n_points, shuffle=True, gen_type="All")

    return etas_train, gs_train


if __name__ == "__main__":
    parser = parse_args()

    params = get_params_from_json(parser.param_file)

    params['device'] = device
    params['layer_prior'] = torch.tensor(params['layer_prior']).to(device)
    params['output_prior'] = torch.tensor(params['output_prior']).to(device)
    params['data_size'] = parser.n_data

    train(parser, params)

    




