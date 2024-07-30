from numpyro.infer import HMC, MCMC

import pickle
import json
import os

def load_numpyro_mcmc(save_dir, file_prefix, mcmc, verbose=True):
    save_path = os.path.join(save_dir, file_prefix)
    i = 0 
    while os.path.exists(save_path + f"_last_state_{i}.pickle"):
        i += 1
    i -= 1

    if verbose:
        print(f"Loading MCMC from {save_path + f'_last_state_{i}.pickle'}")

    mcmc_last_state = pickle.load(open(save_path + f"_last_state_{i}.pickle", "rb"))
    mcmc.post_warmup_state = mcmc_last_state
    return mcmc

def save_numpyro_mcmc(mcmc, save_dir, file_prefix):
    mcmc.sampler= None
    samples = mcmc.get_samples()
    last_state = mcmc.last_state

    save_path = os.path.join(save_dir, file_prefix)
    samples_path = save_path + "_samples_"
    last_state_path = save_path + "_last_state_"

    i = 0
    while True:
        if os.path.exists(samples_path + str(i) + ".pickle") or os.path.exists(last_state_path + str(i) + ".pickle"):
            i += 1
        else:
            break

    pickle.dump(samples, open(samples_path + f"{i}.pickle", "wb"))
    pickle.dump(last_state, open(last_state_path + f"{i}.pickle", "wb"))

def get_params_from_json(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params