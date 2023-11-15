import matplotlib.pyplot as plt
from TurbulenceClosureDataGenerator import TurbulenceClosureDataGenerator
import numpy as np

def plot_data_generation_diff(etas): 

    eta1 = etas[:, 0]
    eta2 = etas[:, 1]

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].scatter(eta1, eta2, marker='.')
    axs[0].set_xlabel("Log(sqrt(eta1))")
    axs[0].set_ylabel("Log(sqrt(eta2))")

    eta1 = (10**eta1)**2
    eta2 = (10**eta2)**2

    axs[1].scatter(eta1, eta2, marker='.')
    axs[1].set_title("Linear Scale")
    axs[1].set_xlabel("eta1")
    axs[1].set_ylabel("eta2")

    plt.show()

def plot_heat_map(model):
    dim = 1_000
    levels = 100
    x, y = np.meshgrid(np.linspace(-.5, 2, dim),np.linspace(-.5, 2, dim))
    eta1 = (10**x.flatten())**2
    eta2 = (10**y.flatten())**2

    gen = TurbulenceClosureDataGenerator(model="SSG", type='torch')
    etas, G_s = gen(eta1, eta2)
    G_s = G_s.T.reshape(3, dim, dim)

    pred = model.predict(etas)

    pred = pred.T.reshape(3, dim, dim)

    fig, axs = plt.subplots(2, 3)

    for (ax0, ax1, ax2), z in zip(axs, [G_s, pred]):
        G_1 = z[0]
        contour_z1 = ax0.contourf(x, y, G_1, levels=levels, cmap='jet')
        ax0.set_title(f"G_1")
        ax0.set_xlabel("log(R)")
        ax0.set_ylabel("log(S)")

        G_2 = z[1]
        contour_z2 = ax1.contourf(x, y, G_2, levels=levels, cmap='jet')
        ax1.set_title(f"G_2")
        ax1.set_xlabel("log(R)")
        ax1.set_ylabel("log(S)")

        G_3 = z[2]
        contour_z3 = ax2.contourf(x, y, G_3, levels=levels, cmap='jet')
        ax2.set_title(f"G_3")
        ax2.set_xlabel("log(R)")
        ax2.set_ylabel("log(S)")

        fig.colorbar(contour_z1)
        fig.colorbar(contour_z2)
        fig.colorbar(contour_z3)
    
    fig.suptitle("SSG Algebraic Reynolds Stress Model (Top Algebraic, Bottom Neural Network)")
    plt.show()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

    error = np.abs(G_s - pred)

    contour_z1 = ax0.contourf(x, y, error[0], levels=levels, cmap='jet')
    ax0.set_title(f"G_1")
    ax0.set_xlabel("log(R)")
    ax0.set_ylabel("log(S)")

    contour_z2 = ax1.contourf(x, y, error[1], levels=levels, cmap='jet')
    ax1.set_title(f"G_2")
    ax1.set_xlabel("log(R)")
    ax1.set_ylabel("log(S)")

    contour_z3 = ax2.contourf(x, y, error[2], levels=levels, cmap='jet')
    ax2.set_title(f"G_3")
    ax2.set_xlabel("log(R)")
    ax2.set_ylabel("log(S)")

    fig.colorbar(contour_z1)
    fig.colorbar(contour_z2)
    fig.colorbar(contour_z3)

    plt.show()