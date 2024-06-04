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

def plot_grid(x_grid, y_grid, z_grids, row_titles, col_titles, frmt, figsize=(20,15), ind_scales = False, save_path=None):
    if not ind_scales:
        g1_min, g1_max = np.min([grid[0] for grid in z_grids]), np.max([grid[0] for grid in z_grids])
        g2_min, g2_max = np.min([grid[1] for grid in z_grids]), np.max([grid[1] for grid in z_grids])
        g3_min, g3_max = np.min([grid[2] for grid in z_grids]), np.max([grid[2] for grid in z_grids])

    fig, axes = plt.subplots(nrows=len(z_grids), ncols=3, figsize=figsize, sharex=True, sharey=True)

    row_font_size = 25
    title_font_size = 20

    for i, (row_axs, z) in enumerate(zip(axes, z_grids)):
        (ax0, ax1, ax2) = row_axs
        if ind_scales:
            g1_min, g1_max = np.min(z[0]), np.max(z[0])
            g2_min, g2_max = np.min(z[1]), np.max(z[1])
            g3_min, g3_max = np.min(z[2]), np.max(z[2])

        G_1 = z[0]
        contour_z1 = ax0.contourf(x_grid, y_grid, G_1, locator=ticker.MaxNLocator(100), cmap='jet', vmin=g1_min, vmax=g1_max)
        #ax0.set_title()
        if i == len(axes) - 1:
            ax0.set_xlabel(r"$\log(\sqrt{\eta_1})$", fontsize=title_font_size)
        ax0.set_ylabel(r"$\log(\sqrt{\eta_2})$", fontsize=title_font_size)
        ax0.tick_params(labelsize=title_font_size)

        G_2 = z[1]
        contour_z2 = ax1.contourf(x_grid, y_grid, G_2, locator=ticker.MaxNLocator(100), cmap='jet', vmin=g2_min, vmax=g2_max)
        #ax1.set_title(f"$G_2$", fontsize=title_font_size)
        if i == len(axes) - 1:
            ax1.set_xlabel(r"$\log(\sqrt{\eta_1})$", fontsize=title_font_size)
        #ax1.set_ylabel(r"$\log(\sqrt{\eta_2})$", fontsize=title_font_size)
        ax1.tick_params(labelsize=title_font_size)


        G_3 = z[2]
        contour_z3 = ax2.contourf(x_grid, y_grid, G_3, locator=ticker.MaxNLocator(100), cmap='jet', vmin=g3_min, vmax=g3_max)
        #ax2.set_title(f"$G_3$", fontsize=title_font_size)
        if i == len(axes) - 1:
            ax2.set_xlabel(r"$\log(\sqrt{\eta_1})$", fontsize=title_font_size)
        #ax2.set_ylabel(r"$\log(\sqrt{\eta_2})$", fontsize=title_font_size)
        ax2.tick_params(labelsize=title_font_size)


        cbar1 = fig.colorbar(contour_z1, format=frmt)
        cbar2 = fig.colorbar(contour_z2, format=frmt)
        cbar3 = fig.colorbar(contour_z3, format=frmt)

        cbar1.ax.tick_params(labelsize=title_font_size)
        cbar1.ax.set_title(f"$G_1$", fontsize=title_font_size)
        cbar2.ax.tick_params(labelsize=title_font_size)
        cbar2.ax.set_title(f"$G_2$", fontsize=title_font_size)
        cbar3.ax.tick_params(labelsize=title_font_size)
        cbar3.ax.set_title(f"$G_3$", fontsize=title_font_size)

    for ax, col in zip(axes[0,:], col_titles):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 22), xycoords='axes fraction', textcoords='offset points', ha='center', va='bottom', size=row_font_size)

    for ax, row in zip(axes[:,0], row_titles):
        ax.annotate(row, xy=(0, 0.5), xytext=(-1, .5), textcoords=ax.yaxis.label, ha='right', va='center', fontsize=row_font_size)


    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()