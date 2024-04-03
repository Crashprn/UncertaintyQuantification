import os
import dill
import json

def create_skorch_model(model, skorch_NN_type, model_args, skorch_args):
    initialized_model = model(**model_args)
    skorch_model = skorch_NN_type(module=initialized_model, **skorch_args)
    return skorch_model

def reinitialize_model(pt_name, checkpoint_dir, model, model_type, net_params, train_params):
    net = create_skorch_model(model, model_type, net_params, train_params)
    net.initialize()
    net.load_params(f_params=os.path.join(checkpoint_dir,pt_name),
                    f_optimizer=os.path.join(checkpoint_dir, 'optimizer.pt'),
                    f_criterion=os.path.join(checkpoint_dir, 'criterion.pt'),
                    f_history=os.path.join(checkpoint_dir, 'history.json')
    )
    return net


def save_MCMC_model(mcmc_obj, save_dir, save_name):
    mcmc_obj.sampler = None

    with open(os.path.join(save_dir, save_name), 'wb') as f:
        dill.dump(mcmc_obj, f)

def load_MCMC_model(save_dir, save_name):
    with open(os.path.join(save_dir, save_name), 'rb') as f:
        mcmc_obj = dill.load(f)
    return mcmc_obj

def get_params_from_json(json_file):
    with open(json_file, 'r') as f:
        params = json.load(f)
    return params
