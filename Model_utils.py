import os

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

