import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker, scale
from TurbulenceClosureDataGenerator import TurbulenceClosureDataGenerator
import numpy as np
import math

def plot_data_generation_diff(etas): 

    eta1 = etas[:, 0]
    eta2 = etas[:, 1]

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].scatter(eta1, eta2, marker='.')
    axs[0].set_xlabel("Log(R)")
    axs[0].set_ylabel("Log(S)")

    eta1 = (10**eta1)
    eta2 = (10**eta2)

    axs[1].scatter(eta1, eta2, marker='.')
    axs[1].set_title("Linear Scale")
    axs[1].set_xlabel("eta1")
    axs[1].set_ylabel("eta2")

    plt.show()

def plot_heat_map_compare(grid_x, grid_y, target, pred):

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    for (ax0, ax1, ax2), z in zip(axs, [target, pred]):
        G_1 = z[0]
        contour_z1 = ax0.contourf(grid_x, grid_y, G_1, locator=ticker.MaxNLocator(100), cmap='jet')
        ax0.set_title(f"G_1")
        ax0.set_xlabel("log(R)")
        ax0.set_ylabel("log(S)")

        G_2 = z[1]
        contour_z2 = ax1.contourf(grid_x, grid_y, G_2, locator=ticker.MaxNLocator(100), cmap='jet')
        ax1.set_title(f"G_2")
        ax1.set_xlabel("log(R)")
        ax1.set_ylabel("log(S)")

        G_3 = z[2]
        contour_z3 = ax2.contourf(grid_x, grid_y, G_3, locator=ticker.MaxNLocator(100), cmap='jet')
        ax2.set_title(f"G_3")
        ax2.set_xlabel("log(R)")
        ax2.set_ylabel("log(S)")

        fig.colorbar(contour_z1)
        fig.colorbar(contour_z2)
        fig.colorbar(contour_z3)
    
    fig.suptitle("SSG Algebraic Reynolds Stress Model (Top Algebraic, Bottom Neural Network)")
    plt.show()

    

def plot_heat_map_loss(x_grid, y_grid, target, pred):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5))

    error = np.abs(target - pred)

    contour_z1 = ax0.contourf(x_grid, y_grid, error[0], locator=ticker.MaxNLocator(100), cmap='jet')
    ax0.set_title(f"G_1")
    ax0.set_xlabel("log(R)")
    ax0.set_ylabel("log(S)")

    contour_z2 = ax1.contourf(x_grid, y_grid, error[1], locator=ticker.MaxNLocator(100), cmap='jet')
    ax1.set_title(f"G_2")
    ax1.set_xlabel("log(R)")
    ax1.set_ylabel("log(S)")

    contour_z3 = ax2.contourf(x_grid, y_grid, error[2], locator=ticker.MaxNLocator(100), cmap='jet')
    ax2.set_title(f"G_3")
    ax2.set_xlabel("log(R)")
    ax2.set_ylabel("log(S)")


    cbar_1 = fig.colorbar(contour_z1)

    cbar_2 = fig.colorbar(contour_z2)

    cbar_3 = fig.colorbar(contour_z3)


    plt.show()

def plot_heat_map_3D(x_grid, y_grid, z_grid):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5))
    levels = 100

    contour_z1 = ax0.contourf(x_grid, y_grid, z_grid[0], locator=ticker.MaxNLocator(100), cmap='jet')
    ax0.set_title(f"G_1")
    ax0.set_xlabel("log(R)")
    ax0.set_ylabel("log(S)")

    contour_z2 = ax1.contourf(x_grid, y_grid, z_grid[1], locator=ticker.MaxNLocator(100), cmap='jet')
    ax1.set_title(f"G_2")
    ax1.set_xlabel("log(R)")
    ax1.set_ylabel("log(S)")

    contour_z3 = ax2.contourf(x_grid, y_grid, z_grid[2], locator=ticker.MaxNLocator(100), cmap='jet')
    ax2.set_title(f"G_3")
    ax2.set_xlabel("log(R)")
    ax2.set_ylabel("log(S)")

    tick = [0.0, 0.0001, 0.002, 0.01, 0.07]

    cbar_1 = fig.colorbar(contour_z1, format="%.4f")
    cbar_1.minorticks_off()

    cbar_2 = fig.colorbar(contour_z2, format="%.4f")

    cbar_3 = fig.colorbar(contour_z3, format="%.4f")

    plt.show()

