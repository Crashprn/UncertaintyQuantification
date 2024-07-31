import argparse

import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, HMC
from numpyro.infer.hmc import hmc
from numpyro.infer.initialization import init_to_value

import jax
import jax.numpy as jnp
from jax import random

import sys
import pickle
import os

sys.path.append(os.path.abspath('.'))

print(sys.path)


from src.models.NumPyroModels import NumPyroModel
from src.data_gens.TurbulenceClosureDataGenerator import TurbulenceClosureDataGenerator
from src.utils.data_utils import *
from src.utils.numpyro_utils import *
from src.utils.data_utils import *


def parse_args():
    parser = argparse.ArgumentParser(
        prog='chpc_main.py',
        description='Train a Bayesian neural network for turbulence closure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--chkpt_dir', '-d', type=str, default='Model_Checkpoints/HMC')
    parser.add_argument('--n_data', type=int, default=2_000)
    parser.add_argument('--verbose', '-v', type=int, default=1)
    parser.add_argument('--param_file', type=str, default='run_params.json')
    parser.add_argument('--init_file', type=str, default='HMC_Initialize_large.pkl')

    return parser.parse_args()

def load_initialization_params(file_path):
    init_dict = pickle.load(open(file_path, "rb"))

    for key, value in init_dict.items():
        init_dict[key] = jnp.array(value.T)

    return init_dict

def find_last_state(save_dir, file_prefix):
    files_in_chkpt_dir = os.listdir(save_dir)
    for file in files_in_chkpt_dir:
        if file.startswith(file_prefix + "_last_state_"):
            if parser.verbose:
                print(f"Found previous state in {parser.chkpt_dir}")
            return True

    return False
    
        

def train(parser, mcmc, save_dir, save_prefix):
    if parser.verbose:
        print(f"Creating {parser.n_data} datapoints")
    etas_train, gs_train = get_data(parser.n_data)

    if parser.verbose:
        print("Scaling Data")
    x_scaler = CustomScalerX().fit(etas_train)
    y_scaler = CustomScalerY().fit(gs_train)

    if parser.verbose:
        print(f"Using devices: {jax.devices()}")
    x = jnp.array(x_scaler.transform(etas_train), dtype=jnp.float32)
    y = jnp.array(y_scaler.transform(gs_train), dtype=jnp.float32)

    if find_last_state(save_dir, save_prefix):
        mcmc = load_numpyro_mcmc(save_dir, save_prefix, mcmc, parser.verbose)
        rng = mcmc.post_warmup_state.rng_key
    else:
        rng = random.PRNGKey(0)
    
    if parser.verbose:
        print("---> Beginning Training")

    mcmc.run(rng, x, y)

    if parser.verbose:
        print("---> Training Complete")
        print(f"---> Saving MCMC object to {save_dir} as {save_prefix}")

    save_numpyro_mcmc(mcmc, save_dir, save_prefix)

    if parser.verbose:
        print("---> Successfully Saved MCMC Object")

    

def get_data(n_points):
    SSG_gen = TurbulenceClosureDataGenerator('SSG')

    # Defining the ranges for the different scales
    log = (-.5, 2)

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

    cwd = os.getcwd()

    save_dir = os.path.join(cwd, parser.chkpt_dir)

    ## Creating Checkpoint Directory if it does not exist
    if not os.path.exists(save_dir):
        if parser.verbose:
            print(f"Creating directory {save_dir}")
        os.makedirs(save_dir)

    ## Checking if parameter file exists if not exit
    param_file_path = os.path.join(save_dir, parser.param_file)
    if not os.path.exists(param_file_path):
        print(f"Could not find parameter file at: {param_file_path}")
        exit(1)
    else:
        if parser.verbose:
            print(f"Loading parameters from {param_file_path}")    
        params = get_params_from_json(os.path.join(save_dir, parser.param_file))

    ## Creating Network
    net_params = params['net_params']
    net_params['data_size'] = parser.n_data
    net = NumPyroModel(**net_params)

    ## Creating HMC object
    hmc_params = params['hmc_params']
    hmc_params['model'] = net

    initialize_file_path = os.path.join(save_dir, parser.init_file) 
    if not os.path.exists(initialize_file_path):
        if parser.verbose:
            print(f"Could not find initialization file at: {initialize_file_path}")
    else:
        if parser.verbose:
            print(f"Loading initialization from {initialize_file_path}")
        net_init_params = pickle.load(open(initialize_file_path, "rb"))
        hmc_params['init_strategy'] = init_to_value(values=load_initialization_params(initialize_file_path))

    hmc = HMC(**hmc_params)

    ## Creating MCMC object

    mcmc_params = params['mcmc_params']

    mcmc = MCMC(hmc, **mcmc_params)

    ## Creating save prefix
    save_prefix = f"HMC_{mcmc_params['num_samples']}_{mcmc_params['num_warmup']}_{int(hmc_params['trajectory_length']/hmc_params['step_size'])}"

    train(parser, mcmc, save_dir, save_prefix)

    




