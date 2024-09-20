from sklearn.preprocessing import StandardScaler

import torch

import argparse

import os
import sys
import pickle

sys.path.append(os.path.abspath('.'))

from src.data_gens.TurbulenceClosureDataGenerator import TurbulenceClosureDataGenerator
from src.models.GaussianProcess import GaussianProcessRegressor, RBFKernel

from src.utils.model_utils import *
from src.utils.data_utils import *


DATA_BOUNDS_LOG = (-.5, 2)
#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
DTYPE = torch.float64

def parse_args():
    parser = argparse.ArgumentParser(
        prog='chpc_main.py',
        description='Train a Bayesian neural network for turbulence closure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--save_dir', '-d', type=str, default='data/GP')
    parser.add_argument('--n_data', type=int, default=1_000)
    parser.add_argument('--grid_dim', type=int, default=700)
    parser.add_argument('--verbose', '-v', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--dim_y', type=int, default=0)
    parser.add_argument('--run_name', type=str, default='GP')

    return parser.parse_args()

def train_test(parser):

    if parser.verbose:
        print(f"Creating {parser.n_data} datapoints")
    etas_train, gs_train = get_data(parser.n_data)
    gs_train = gs_train[:, parser.dim_y].reshape(-1, 1)

    if parser.verbose:
        print("Scaling Data")

    x_scaler = CustomScalerX().fit(etas_train)
    y_scaler = StandardScaler().fit(gs_train)
    x_train = x_scaler.transform(etas_train).astype(np.float32)
    y_train = y_scaler.transform(gs_train).astype(np.float32)

    x_train = torch.tensor(x_train, device=DEVICE, dtype=DTYPE)
    y_train = torch.tensor(y_train, device=DEVICE, dtype=DTYPE)


    kernel = RBFKernel(amplitude=6.0, length_scale=0.001, device=DEVICE).to(DEVICE)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        batch_size=parser.batch_size,
        device=DEVICE,
        verbose=parser.verbose,
        max_iter=1000,
        lr=1e-3,
        delta=1e-6
    )

    if parser.verbose:
        print("---> Fitting Gaussian Process")

    gp = gp.fit(x_train, y_train)

    if parser.verbose:
        print("---> Fit Complete")
        print("---> Generating Predictions")

    save_dir = parser.save_dir
    os.makedirs(save_dir, exist_ok=True)

    param_dict = gp.kernel.get_params()

    if parser.verbose:
        print("---> Saving Kernel Hyperparameters")
        print(f"Kernel Hyperparameters: {param_dict}")
    
    pickle.dump(param_dict, open(os.path.join(save_dir, f"{parser.run_name}_{parser.dim_y}_KernelParams.pkl"), "wb"))
    
    etas_test, gs_test = get_test_data(parser)

    etas_test = x_scaler.transform(etas_test)

    pred_mean = []
    pred_std = []

    num_splits = 100

    for x_split in np.array_split(etas_test, num_splits):
        x_split = torch.tensor(x_split, device=DEVICE, dtype=DTYPE)
        mean, std = gp.predict(x_split, return_std=True)
        pred_mean.append(y_scaler.inverse_transform(mean.reshape(-1, 1).detach().cpu().numpy()))
        pred_std.append(std.detach().cpu().numpy() * y_scaler.scale_)
    
    pred_mean = np.concatenate(pred_mean)
    pred_std = np.concatenate(pred_std)

    if parser.verbose:
        print("---> Predictions Complete")
        print("---> Saving Predictions to ", save_dir)
        target = gs_test[:, parser.dim_y]
        print("---> Test Error: ", np.mean(np.abs(target - pred_mean.squeeze())))

    np.save(os.path.join(save_dir, f"{parser.run_name}_Mean{parser.dim_y}.npy"), pred_mean)
    np.save(os.path.join(save_dir, f"{parser.run_name}_Std{parser.dim_y}.npy"), pred_std)

    if parser.verbose:
        print("---> Predictions Saved")

def get_test_data(parser):
    dim = parser.grid_dim
    x_grid, y_grid = np.meshgrid(np.linspace(*DATA_BOUNDS_LOG, dim),np.linspace(*DATA_BOUNDS_LOG, dim))
    eta1 = (10**x_grid.flatten())**2
    eta2 = (10**y_grid.flatten())**2
    
    if parser.verbose:
        print("---> Generating Test Data")

    gen = TurbulenceClosureDataGenerator(model="SSG", type='numpy')

    if parser.verbose:
        print("---> Finished Generating Test Data")

    etas, G_s = gen(eta1, eta2)

    return etas, G_s

def get_data(n_points):
    SSG_gen = TurbulenceClosureDataGenerator('SSG')

    # Defining whether to exclude certain areas of the data
    exclude_area = False
    include_area = False
    drop_eta_1 = False
    drop_eta_2 = False
    add_noise = False

    # Defining area to exclude datapoints
    eta_1_range = (10**np.array([-.3, 0.0]))**2
    eta_2_range = (10**np.array([-.3, 0.0]))**2

    etas_train, gs_train = generate_log_data(SSG_gen, DATA_BOUNDS_LOG, n_points, shuffle=True, gen_type="d_condition", d_condition='>=')

    return etas_train, gs_train


if __name__ == "__main__":
    parser = parse_args()

    train_test(parser)
