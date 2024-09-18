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
    parser.add_argument('--n__train_data', type=int, default=1_000)
    parser.add_argument('--vis_data_dir', type=int, default='data/vis_data')
    parser.add_argument('--verbose', '-v', type=int, default=1)
    parser.add_argument('--dim_y', type=int, default=0)
    parser.add_argument('--run_name', type=str, default='GP')

    return parser.parse_args()

def get_gp_samples(parser):

    if parser.verbose:
        print("Loading Kernel Hyperparameters")
    param_dict = pickle.load(open(os.path.join(parser.save_dir, f"{parser.run_name}_{parser.dim_y}_KernelParams.pkl"), "rb"))

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


    kernel = RBFKernel(amplitude=param_dict['amplitude'], length_scale=param_dict['length_scale'], device=DEVICE).to(DEVICE)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts=parser.n_restarts,
        batch_size=parser.batch_size,
        device=DEVICE,
        verbose=parser.verbose,
        max_iter=0,
        lr=1e-3,
        delta=1e-6
    )

    if parser.verbose:
        print("---> Initializing Gaussian Process")

    gp = gp.fit(x_train, y_train)

    if parser.verbose:
        print("---> Generating Predictions")
    
    save_dir = parser.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if parser.verbose:
        print("---> Loading Test Data")
    
    x_test = np.load(os.path.join(parser.vis_data_dir, f"{parser.run_name}_Test_Data.npy"))


    x_test = x_scaler.transform(x_test)

    x_test = torch.tensor(x_test, device=DEVICE, dtype=DTYPE)
    samples = gp.predict(x_test, num_samples=100)
    
    if parser.verbose:
        print("---> Predictions Complete")
        print("---> Saving Predictions to ", save_dir)

    np.save(os.path.join(save_dir, f"{parser.run_name}_Samples.npy"), samples.cpu().detach().numpy())

    if parser.verbose:
        print("---> Predictions Saved")

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

    etas_train, gs_train = generate_log_data(SSG_gen, DATA_BOUNDS_LOG, n_points, shuffle=True, gen_type="All", d_condition='>=')

    return etas_train, gs_train


if __name__ == "__main__":
    parser = parse_args()

    get_gp_samples(parser)
