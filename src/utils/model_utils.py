import os
import dill
import pickle
import json
import pyro.infer
import torch
import pyro

'''
Function to create a Skorch model.
Parameters:
model: torch.nn.Module
    The PyTorch model to be wrapped in Skorch.
skorch_NN_type: skorch.NeuralNet
    The Skorch NeuralNet class to be used.
model_args: dict
    Arguments to initialize the PyTorch model.
skorch_args: dict
    Arguments to initialize the Skorch NeuralNet.

Returns:
skorch_model: skorch.NeuralNet
    The Skorch model initialized with the given parameters.
'''
def create_skorch_model(model, skorch_NN_type, model_args, skorch_args):
    initialized_model = model(**model_args)
    skorch_model = skorch_NN_type(module=initialized_model, **skorch_args)
    skorch_model.initialize()
    return skorch_model

'''
Function to reinitialize a model with the given parameters.
Parameters:
pt_name: str
    Path to the parameters file.
net: torch.nn.Module
    The model to be reinitialized.

Returns:
net: torch.nn.Module
    The reinitialized model with loaded parameters.
'''
def reinitialize_model(pt_name, net):
    net.initialize()
    net.load_params(f_params=pt_name)
    return net


def save_MCMC_model(mcmc_obj, save_dir, save_name):
    mcmc_obj.sampler = None

    with open(os.path.join(save_dir, save_name), 'wb') as f:
        dill.dump(mcmc_obj._samples, f)

def load_MCMC_model(save_dir, save_name, kernel, sample_params):
    mcmc_obj = pyro.infer.MCMC(kernel, **sample_params)
    with open(os.path.join(save_dir, save_name), 'rb') as f:
        mcmc_obj._samples = dill.load(f)

    mcmc_obj._diagnostics = [{'divergences':[], 'acceptance rate': 0.00} for _ in mcmc_obj._diagnostics]

    return mcmc_obj

'''
Funciton to fetch parameters from a JSON file.
Parameters:
json_file: str
    Path to the JSON file containing parameters.
Returns:
params: dict
    Dictionary containing parameters loaded from the JSON file.
'''
def get_params_from_json(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params

'''
Function to save a Pyro model and its guide to a specified path.
Parameters:
model: Pyro model
    The Pyro model to be saved.
guide: Pyro guide
    The Pyro guide to be saved.
path: str
    Path to the directory where the model and guide will be saved.
chkpt_name: str
    Name of the checkpoint files (default is "pyro").

Returns:
None
'''
def save_pyro_model(model, guide, path, chkpt_name="pyro"):
    torch.save({"model": model.state_dict(), "guide": guide}, os.path.join(path, chkpt_name+"_model.pt"))
    pyro.get_param_store().save(os.path.join(path, chkpt_name+"_params.pt"))

'''
Function to load a Pyro model and its guide from a specified path.
Parameters:
model: Pyro model
    The Pyro model to be loaded.
path: str
    Path to the directory where the model and guide are saved.
device: str
    Device to load the model onto (e.g., 'cpu' or 'cuda').
chkpt_name: str
    Name of the checkpoint files (default is "pyro").
'''
def load_pyro_model(model, path, device, chkpt_name="pyro"):
    model_path = os.path.join(path, chkpt_name+"_model.pt")
    saved_model_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(saved_model_dict["model"])
    guide = saved_model_dict["guide"]
    pyro.get_param_store().load(os.path.join(path, chkpt_name+"_params.pt"), device)
    pyro.module('module', model, update_module_params=True)
    pyro.module('guide', guide, update_module_params=True)
    return model, guide

'''
Function to convert PyTorch model parameters to numpy arrays and save them as a pickle file.
Parameters are saved in a dictionary format where keys are 'w1', 'b1', etc. for weights and biases respectively.
Parameters:
path: str
    Path to the PyTorch model file (.pt) to be converted.
Returns:
None
'''
def convert_torch_to_numpy(path):
    tensors = torch.load(path) 
    param_dict = {k: v.detach().cpu().numpy() for k, v in tensors.items()}
    new_dict = {}
    i = 1
    for k, v in param_dict.items():
        desc = k.split(".")
        if 'weight' in desc[-1]:
            new_dict["w"+str(i)] = v
        else:
            new_dict["b"+str(i)] = v
            i += 1


    pickle.dump(new_dict, open(path.replace(".pt", ".pkl"), "wb"))