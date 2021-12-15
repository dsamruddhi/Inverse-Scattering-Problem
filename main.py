import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from utils.plot_utils import PlotUtils

from scatterer.scatterer import Scatterer
from forward_problem.solve import ForwardProblemSolver
from inverse_problem.solve import InverseProblemSolver

if __name__ == '__main__':

##
    """" Define scatterer """

    scatterer_params = [
    {
        "shape": "circle",
        "center_x": -0.3,
        "center_y": 0.35,
        "size": 0.15,
        "permittivity": 4 + 0.4j
    },
    {

        "shape": "circle",
        "center_x": 0.3,
        "center_y": 0.35,
        "size": 0.15,
        "permittivity": 4 + 0.4j
    }
    ]

    """" Generate and save scatterer """

    scatterer = Scatterer("forward", scatterer_params).generate()
    scatterer1 = Scatterer("inverse", scatterer_params).generate()
    savemat("data/scatterer.mat", {"scatterer": scatterer})
    savemat("data/scatterer_inverse.mat", {"scatterer": scatterer1})

    fig1, (ax1, ax2) = plt.subplots(ncols=2)
    original = ax1.imshow(np.real(scatterer), cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig1.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
    ax1.title.set_text("Scatterer: Real component")
    reconstructed_real = ax2.imshow(np.imag(scatterer), cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig1.colorbar(reconstructed_real, ax=ax2, fraction=0.046, pad=0.04)
    ax2.title.set_text("Scatterer: Imaginary component")
    plt.savefig("data/scatterer.png")
    plt.show()

##
    """" Solve Forward Problem """

    forward_solver = ForwardProblemSolver(scatterer)
    direct_field, direct_power, scattered_field, total_field, total_power = forward_solver.generate_forward_data()
    ForwardProblemSolver.get_field_plots(total_field, direct_field, scattered_field, 39)
    ForwardProblemSolver.save_data(scatterer, direct_field, direct_power, scattered_field, total_field, total_power)

##
    """" Solve Inverse Problem """

    model_name = "prytov_imag"
    prior = "qs2D"
    params = {"alpha": 15, "sparse": True}
    inverse_solver = InverseProblemSolver(direct_power, total_power, model_name, prior, params)

    if model_name == "prytov_imag":
        imag_rec = inverse_solver.solve()
        real_rec = np.zeros(imag_rec.shape)
    else:
        real_rec, imag_rec = inverse_solver.solve()

    """ Plot reconstruction """

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # fig.tight_layout()

    original_real = ax1.imshow(np.real(scatterer), cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
    cb1 = fig.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
    cb1.ax.tick_params(labelsize=12)
    ax1.title.set_text(f"Original scatterer (real)")
    ax1.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

    guess_real = ax2.imshow(real_rec, cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
    cb2 = fig.colorbar(guess_real, ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.tick_params(labelsize=12)
    ax2.title.set_text("Initial guess: Real component")
    ax2.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

    guess_imag = ax3.imshow(imag_rec, cmap=plt.cm.jet, extent=[-0.75, 0.75, -0.75, 0.75])
    cb3 = fig.colorbar(guess_imag, ax=ax3, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=12)
    ax3.title.set_text("Initial guess: Imaginary component")
    ax3.set(xticks=[-0.75, 0, 0.75], yticks=[-0.75, 0, 0.75])

    plt.setp(ax1.get_xticklabels(), fontsize=12, horizontalalignment="left")
    plt.setp(ax2.get_xticklabels(), fontsize=12, horizontalalignment="left")
    plt.setp(ax3.get_xticklabels(), fontsize=12, horizontalalignment="left")

    plt.setp(ax1.get_yticklabels(), fontsize=12)
    plt.setp(ax2.get_yticklabels(), fontsize=12)
    plt.setp(ax3.get_yticklabels(), fontsize=12)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
    plt.show()

    savemat("data/input.mat", {"real_rec": real_rec, "imag_rec": imag_rec})

    """ Save results """
    savemat("data/real_rec.mat", {"real_rec": real_rec})
    savemat("data/imag_rec.mat", {"imag_rec": imag_rec})
