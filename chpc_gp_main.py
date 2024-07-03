from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import argparse

import os

from src.models.TurbulenceNetwork import TurbulenceNetworkBayesian
from src.data_gens.TurbulenceClosureDataGenerator import TurbulenceClosureDataGenerator

from src.utils.model_utils import *
from src.utils.data_utils import *


DATA_BOUNDS_LOG = (-.5, 2)

def parse_args():
    parser = argparse.ArgumentParser(
        prog='chpc_main.py',
        description='Train a Bayesian neural network for turbulence closure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--save_dir', '-d', type=str, default='data')
    parser.add_argument('--n_data', type=int, default=5_000)
    parser.add_argument('--n_restarts', type=int, default=10)
    parser.add_argument('--grid_dim', type=int, default=700)
    parser.add_argument('--verbose', '-v', type=int, default=1)
    parser.add_argument('--dim_y', type=int, default=0)

    return parser.parse_args()

def train_test(parser):

    if parser.verbose:
        print(f"Creating {parser.n_data} datapoints")
    etas_train, gs_train = get_data(parser.n_data)

    if parser.verbose:
        print("Scaling Data")

    x_scaler = CustomScalerX().fit(etas_train)
    y_scaler = StandardScaler().fit(gs_train)
    x_train = x_scaler.transform(etas_train).astype(np.float32)
    y_train = y_scaler.transform(gs_train).astype(np.float32)[:,parser.dim_y]

    kernel = 1.0*RBF(length_scale=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=parser.n_restarts)

    if parser.verbose:
        print("---> Fitting Gaussian Process")

    gp = gp.fit(x_train, y_train)

    if parser.verbose:
        print("---> Fit Complete")
        print("---> Generating Predictions")

    etas_test, gs_test = get_test_data(parser)

    etas_test = x_scaler.transform(etas_test)

    pred_mean, pred_std = gp.predict(etas_test, return_std=True)

    pred_mean = y_scaler.inverse_transform(pred_mean)
    pred_std = pred_std * y_scaler.scale_

    if parser.verbose:
        print("---> Predictions Complete")
        print("---> Saving Predictions")
    
    save_dir = os.path.join(parser.save_dir, "GP")
    os.makedirs(save_dir, exist_ok=True)

    if parser.verbose:
        print("---> Saving Predictions to ", save_dir)

    np.save(os.path.join(save_dir, f"Mean{parser.dim_y}.npy"), pred_mean)
    np.save(os.path.join(save_dir, f"Std{parser.dim_y}.npy"), pred_std)

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

    etas_train, gs_train = generate_log_data(SSG_gen, DATA_BOUNDS_LOG, n_points, shuffle=True, gen_type="All")

    return etas_train, gs_train


if __name__ == "__main__":
    parser = parse_args()

    print(parser.verbose)

    train_test(parser)
