import argparse
import pickle
import os
import sys

import numpy as np
import numpyro
from numpyro import distributions as dist
from numpyro.infer.hmc import hmc
from numpyro.infer import HMC
from numpyro.infer.initialization import init_to_value, init_to_uniform
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect

import jax
import jax.numpy as jnp
from jax import random

sys.path.append(os.path.abspath('.'))

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
    parser.add_argument('--n_data', type=int, default=80_000)
    parser.add_argument('--verbose', '-v', type=int, default=1)
    parser.add_argument('--param_file', type=str, default='run_params.json')
    parser.add_argument('--init_file', type=str, default='HMC_Initialize_large.pkl')

    return parser.parse_args()

def load_initialization_params(file_path):
    init_dict = pickle.load(open(file_path, "rb"))

    for key, value in init_dict.items():
        init_dict[key] = jnp.array(value.T)

    return init_dict

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

def get_diagnostics_str(state):
    return "{} steps of size {:.2e}. acc. prob={:.2f}".format(
        int(state.trajectory_length/state.adapt_state.step_size), state.adapt_state.step_size, state.mean_accept_prob
    )

def run_warmup(parser, model, init_strat, hmc_params, mcmc_params, warmup_iters, save_dir, save_prefix):
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

    init_rng_key, sample_rng_key = random.split(random.PRNGKey(0))
    inverse_mass_matrix = None
    model_info = initialize_model(init_rng_key, model, init_strategy=init_strat, model_args=(x, y))
    init_params = model_info.param_info

    init_kernel, sample_kernel = hmc(model_info.potential_fn, algo="HMC")

    ## Loading Previous warmup state if it exists
    if os.path.exists(os.path.join(save_dir, f"{save_prefix}_warm_state.pkl")):
        if parser.verbose:
            print(f"Found Warmup State {save_prefix}_warm_state.pkl at {save_dir}")
        hmc_state = pickle.load(open(os.path.join(save_dir, f"{save_prefix}_warm_state.pkl"), "rb"))
        ## Reinitializing kernel because of global variables
        _ = init_kernel(
            init_params=hmc_state.z,
            num_warmup=mcmc_params['num_warmup'],
            rng_key=hmc_state.rng_key,
            inverse_mass_matrix=hmc_state.adapt_state.inverse_mass_matrix,
            **hmc_params,
            num_steps=int(hmc_params['trajectory_length']/hmc_params['step_size'])
        )
    else:
        hmc_state = init_kernel(
            init_params=init_params, 
            num_warmup=mcmc_params['num_warmup'],
            rng_key=sample_rng_key,
            **hmc_params,
            num_steps=int(hmc_params['trajectory_length']/hmc_params['step_size'])
        )

    remaining_iterations = mcmc_params['num_warmup'] - hmc_state.i

    num_iters = warmup_iters if warmup_iters < remaining_iterations else remaining_iterations

    if num_iters <= 0:
        print(f"Already completed {mcmc_params['num_warmup']} warmup iterations")
        return

    diagnostics = (
            lambda x: get_diagnostics_str(x)
    )

    hmc_states = fori_collect(
        num_iters - 1,
        num_iters,
        sample_kernel,
        hmc_state,
        diagnostics_fn=diagnostics,
        transform=lambda hmc_state: model_info.postprocess_fn(hmc_state.z),
        return_last_val=True
    )

    if parser.verbose:
        print(f"Saving Warmup State to {save_dir} as {save_prefix}_warm_state.pkl")
    pickle.dump(hmc_states[1], open(os.path.join(save_dir, f"{save_prefix}_warm_state.pkl"), "wb"))

    if parser.verbose:
        if  remaining_iterations < warmup_iters:
            print(f"Finished Warmup after {num_iters} iterations")
        else:
            print(f"Finished {warmup_iters} warmup iterations, {remaining_iterations-warmup_iters} remaining")


if __name__ == '__main__':
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

    initialize_file_path = os.path.join(save_dir, parser.init_file) 
    if not os.path.exists(initialize_file_path):
        if parser.verbose:
            print(f"Could not find initialization file at: {initialize_file_path}")
        init_strat = init_to_uniform
    else:
        if parser.verbose:
            print(f"Loading initialization from {initialize_file_path}")
        net_init_params = pickle.load(open(initialize_file_path, "rb"))
        init_strat = init_to_value(values=load_initialization_params(initialize_file_path))

    mcmc_params = params['mcmc_params']

    ## Creating save prefix
    save_prefix = f"HMC_{mcmc_params['num_samples']}_{mcmc_params['num_warmup']}_{int(hmc_params['trajectory_length']/hmc_params['step_size'])}"

    run_warmup(parser, net, init_strat, hmc_params, mcmc_params, params['warmup_iter_block'], save_dir, save_prefix)



