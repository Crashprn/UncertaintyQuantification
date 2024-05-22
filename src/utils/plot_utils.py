import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import ticker, scale
from src.data_gens.TurbulenceClosureDataGenerator import TurbulenceClosureDataGenerator
import numpy as np
import math

plt.rcParams['text.usetex'] = True


def plot_data_generation_diff(etas): 

    eta1 = etas[:, 0]
    eta2 = etas[:, 1]

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    axs.scatter(np.log10(np.sqrt(eta1)), np.log10(np.sqrt(eta2)), marker='.')
    axs.set_xlabel(r"$\log(\sqrt{\eta_1})$")
    axs.set_ylabel(r"$\log(\sqrt{\eta_2})$")

    plt.show()

def plot_heat_map_compare(grid_x, grid_y, target, pred, top_title="Algebraic", bottom_title="Neural Network", sup_title=True):

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    g1_min = min(np.min(target[0]), np.min(pred[0]))
    g1_max = max(np.max(target[0]), np.max(pred[0]))

    g2_min = min(np.min(target[1]), np.min(pred[1]))
    g2_max = max(np.max(target[1]), np.max(pred[1]))

    g3_min = min(np.min(target[2]), np.min(pred[2]))
    g3_max = max(np.max(target[2]), np.max(pred[2]))

    for (ax0, ax1, ax2), z in zip(axs, [target, pred]):
        G_1 = z[0]
        contour_z1 = ax0.contourf(grid_x, grid_y, G_1, locator=ticker.MaxNLocator(100), cmap='jet', vmin=g1_min, vmax=g1_max)
        ax0.set_title(f"$G_1$")
        ax0.set_xlabel(r"$\log(\sqrt{\eta_1})$")
        ax0.set_ylabel(r"$\log(\sqrt{\eta_2})$")

        G_2 = z[1]
        contour_z2 = ax1.contourf(grid_x, grid_y, G_2, locator=ticker.MaxNLocator(100), cmap='jet', vmin=g2_min, vmax=g2_max)
        ax1.set_title(f"$G_2$")
        ax1.set_xlabel(r"$\log(\sqrt{\eta_1})$")
        ax1.set_ylabel(r"$\log(\sqrt{\eta_2})$")

        G_3 = z[2]
        contour_z3 = ax2.contourf(grid_x, grid_y, G_3, locator=ticker.MaxNLocator(100), cmap='jet', vmin=g3_min, vmax=g3_max)
        ax2.set_title(f"$G_3$")
        ax2.set_xlabel(r"$\log(\sqrt{\eta_1})$")
        ax2.set_ylabel(r"$\log(\sqrt{\eta_2})$")

        fig.colorbar(contour_z1)
        fig.colorbar(contour_z2)
        fig.colorbar(contour_z3)
    
    if sup_title:
        fig.suptitle(f"SSG Algebraic Reynolds Stress Model (Top {top_title}, Bottom {bottom_title})")
    plt.show()

    

def plot_heat_map_loss(x_grid, y_grid, target, pred):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5))

    error = np.abs(target - pred)

    contour_z1 = ax0.contourf(x_grid, y_grid, error[0], locator=ticker.MaxNLocator(100), cmap='jet')
    ax0.set_title(f"$G_1$")
    ax0.set_xlabel(r"$\log(\sqrt{\eta_1})$")
    ax0.set_ylabel(r"$\log(\sqrt{\eta_2})$")

    contour_z2 = ax1.contourf(x_grid, y_grid, error[1], locator=ticker.MaxNLocator(100), cmap='jet')
    ax1.set_title(f"$G_2$")
    ax1.set_xlabel(r"$\log(\sqrt{\eta_1})$")
    ax1.set_ylabel(r"$\log(\sqrt{\eta_2})$")

    contour_z3 = ax2.contourf(x_grid, y_grid, error[2], locator=ticker.MaxNLocator(100), cmap='jet')
    ax2.set_title(f"$G_3$")
    ax2.set_xlabel(r"$\log(\sqrt{\eta_1})$")
    ax2.set_ylabel(r"$\log(\sqrt{\eta_2})$")

    cbar_1 = fig.colorbar(contour_z1)

    cbar_2 = fig.colorbar(contour_z2)

    cbar_3 = fig.colorbar(contour_z3)


    plt.show()

def plot_heat_map_3D(x_grid, y_grid, z_grid, title="", sup_title=True):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 5))

    contour_z1 = ax0.contourf(x_grid, y_grid, z_grid[0], locator=ticker.MaxNLocator(100), cmap='jet')
    ax0.set_title(f"$G_1$")
    ax0.set_xlabel(r"$\log(\sqrt{\eta_1})$")
    ax0.set_ylabel(r"$\log(\sqrt{\eta_2})$")

    contour_z2 = ax1.contourf(x_grid, y_grid, z_grid[1], locator=ticker.MaxNLocator(100), cmap='jet')
    ax1.set_title(f"$G_2$")
    ax1.set_xlabel(r"$\log(\sqrt{\eta_1})$")
    ax1.set_ylabel(r"$\log(\sqrt{\eta_2})$")

    contour_z3 = ax2.contourf(x_grid, y_grid, z_grid[2], locator=ticker.MaxNLocator(100), cmap='jet')
    ax2.set_title(f"$G_3$")
    ax2.set_xlabel(r"$\log(\sqrt{\eta_1})$")
    ax2.set_ylabel(r"$\log(\sqrt{\eta_2})$")

    cbar_1 = fig.colorbar(contour_z1, format="%.4f")

    cbar_2 = fig.colorbar(contour_z2, format="%.4f")

    cbar_3 = fig.colorbar(contour_z3, format="%.4f")

    
    if sup_title:
        fig.suptitle(title)

    plt.show()

