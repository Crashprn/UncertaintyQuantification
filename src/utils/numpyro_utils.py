import jax.numpy as jnp

import pickle
import json
import os

def load_numpyro_mcmc(save_path, verbose=True):
    if verbose:
        print(f"Loading MCMC from {save_path}")

    mcmc_last_state = pickle.load(open(save_path, "rb"))
    return mcmc_last_state

def save_numpyro_mcmc(mcmc, save_dir, file_prefix):
    samples = mcmc.get_samples(group_by_chain=mcmc.num_chains > 1)
    last_state = mcmc.last_state

    save_path = os.path.join(save_dir, file_prefix)
    samples_path = save_path + "_samples.pkl"
    last_state_path = save_path + "_last_state.pkl"

    if os.path.exists(samples_path):
        old_samples = pickle.load(open(samples_path, "rb"))
        for key in samples.keys():
            if len(samples[key].shape) != len(old_samples[key].shape):
                print(f"Shape mismatch for {key}, old samples have shape {old_samples[key].shape} and new samples have shape {samples[key].shape} saves have incompatible number of chains")
            if mcmc.num_chains > 1:
                samples[key] = jnp.concatenate([old_samples[key], samples[key]], axis=1)
            else:
                samples[key] = jnp.concatenate([old_samples[key], samples[key]], axis=0)

    pickle.dump(samples, open(samples_path, "wb"))
    pickle.dump(last_state, open(last_state_path, "wb"))

def get_params_from_json(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params