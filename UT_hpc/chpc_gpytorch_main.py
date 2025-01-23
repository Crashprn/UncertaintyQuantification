# %%
## Standard GP
import numpy as np
import gpytorch
import math
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

import os
import sys

sys.path.append(os.path.abspath('.'))

from src.models.Gpytorch_GP.batchedGPVariants import GPModel
from src.models.Gpytorch_GP.standardGP import standardGP
from src.utils.data_utils import generate_log_data, CustomScalerX
from sklearn.preprocessing import StandardScaler
from src.data_gens.TurbulenceClosureDataGenerator import TurbulenceClosureDataGenerator

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog='chpc_main.py',
        description='Train a GPytorch GP for turbulence closure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--save_dir', '-d', type=str, default='data/GP/GP_TEST/GP1')
    parser.add_argument('--n_data', type=int, default=80_000)
    parser.add_argument('--grid_dim', type=int, default=700)
    parser.add_argument('--verbose', '-v', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--n_inducing', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=300)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--run_name', type=str, default='Regular')
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--sample_dir', type=str, default='data/')

    return parser.parse_args()

if __name__ == "__main__":
    parser = parse_args()

    LOG = (-0.5, 2.0)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    dtype = torch.float

    gen = TurbulenceClosureDataGenerator(model='SSG')
    train_x, train_y = generate_log_data(gen, LOG, parser.n_data, shuffle=True, gen_type="All", d_condition="<=", noise_type="out_noise", noise=0.001**2)

    x_transform_obj = CustomScalerX().fit(train_x)
    y_transform_obj = StandardScaler().fit(train_y[:, parser.y_dim].reshape(-1,1))
    train_x_norm = torch.tensor(x_transform_obj.transform(train_x), dtype=dtype).to(device)
    train_y_norm = torch.tensor(y_transform_obj.transform(train_y[:, parser.y_dim].reshape(-1,1)), dtype=dtype).to(device)

    BATCH_SIZE = parser.batch_size
    NUM_EPOCHS = parser.max_iter if not parser.sample else 0
    NUM_INDUCING = parser.n_inducing
    NUM_DIM = train_x_norm.shape[1]
    INDUCING_PTS = (-1+2*torch.rand((NUM_INDUCING, NUM_DIM))).to(device) # train_x_norm # 
    BATCHES = 1


    train_x_norm = train_x_norm.repeat(BATCHES,1,1)
    train_y_norm = train_y_norm.squeeze().repeat(BATCHES,1)

    sgp = standardGP(num_inducing=NUM_INDUCING, 
                    initial_inducing_pts=INDUCING_PTS, 
                    learn_inducing=True, 
                    num_dim=NUM_DIM, num_GPs=BATCHES, 
                    train_inp=train_x_norm, train_out=train_y_norm,
                    device=device)

    if parser.verbose and parser.resume:
        print("Resuming Training of the GP model")

    if parser.resume or parser.sample:
        model_params = torch.load(os.path.join(parser.save_dir, f"{parser.run_name}_{parser.y_dim}_Params.pt"))
    else:
        model_params = None

    trained_model = sgp.train(epochs=NUM_EPOCHS,learning_rate=0.01, model_params=model_params)

    ##### Save the model
    save_dir = parser.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if not parser.sample:

        if parser.verbose:
            print(f"Saving model to {save_dir}")
        torch.save(trained_model.state_dict(), os.path.join(save_dir, f"{parser.run_name}_{parser.y_dim}_Params.pt"))


    if parser.sample:
        sample_x = np.load(os.path.join(parser.sample_dir, "etas.npy"))
        sample_y = np.load(os.path.join(parser.sample_dir, "gs.npy"))
        sample_y = sample_y[:, parser.y_dim].reshape(1, -1)
        sample_x = x_transform_obj.transform(sample_x)
        sample_x = torch.tensor(sample_x, dtype=dtype, device=device)
        print("Sampling from the model")
        samples = sgp.predict(sample_x, trained_model, batch_size=100, n_samples=parser.n_samples).squeeze().cpu().detach().numpy()

        np.save(os.path.join(save_dir, f"{parser.run_name}_{parser.y_dim}_Samples.npy"), samples)

        sys.exit(0)

    ##### Testing

    if parser.verbose:
        print("Generating test data")

    dim = parser.grid_dim
    x_grid, y_grid = np.meshgrid(np.linspace(*LOG, dim),np.linspace(*LOG, dim))
    eta1 = (10**x_grid.flatten())**2
    eta2 = (10**y_grid.flatten())**2
    test_x, test_y = gen(eta1, eta2)

    test_x_norm = torch.tensor(x_transform_obj.transform(test_x), dtype=dtype, device=device)
    test_y_norm = test_y[:, parser.y_dim].reshape(-1,1)

    if parser.verbose:
        print("Getting predictions for test data")

    predictive_means, predictive_variances = sgp.predict(test_x_norm, trained_model, batch_size=50)

    predictive_means = predictive_means.cpu().detach().numpy()
    predictive_variances = predictive_variances.cpu().detach().numpy()

    predictive_means_real = y_transform_obj.inverse_transform(predictive_means)
    predictive_variances_real = predictive_variances*y_transform_obj.scale_

    if parser.verbose:
        print(f"Mean Absolute Error: {np.abs(test_y_norm.squeeze() - predictive_means_real.squeeze()).mean()}")
        print("Saving predictions to ", save_dir + "/" + parser.run_name)

    np.save(os.path.join(save_dir, f"{parser.run_name}_Mean{parser.y_dim}.npy"), predictive_means_real)
    np.save(os.path.join(save_dir, f"{parser.run_name}_Std{parser.y_dim}.npy"), predictive_variances_real)

# %%
