import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from utils.doi_utils import DOIUtils
from utils.plot_utils import PlotUtils

from forward_problem.solve import ForwardProblemSolver
from inverse_problem.solve import InverseProblemSolver

if __name__ == '__main__':

    """" Define scatterer """

    action = "read"

    if action == "read":
        scatterer = loadmat("data/scatterer.mat")["scatterer"]

    else:
        grid_positions = DOIUtils.get_grid_centroids("forward")
        scatterer_params = {
            "shape": "circle",
            "center_x": -0.3,
            "center_y": 0.5,
            "size": 0.15,  # radius for circle, side for square,
            "permittivity": 4
        }

        m = len(grid_positions[0])
        scatterer = np.ones((m, m), dtype=float)

        if scatterer_params["shape"] == "circle":
            scatterer[(grid_positions[0] - scatterer_params["center_x"]) ** 2 +
                      (grid_positions[1] - scatterer_params["center_y"]) ** 2
                      <= scatterer_params["size"] ** 2] = scatterer_params["permittivity"]

        elif scatterer_params["shape"] == "square":
            mask = ((grid_positions[0] <= scatterer_params["center_x"] + scatterer_params["size"]) &
                    (grid_positions[0] >= scatterer_params["center_x"] - scatterer_params["size"]) &

                    (grid_positions[1] <= scatterer_params["center_y"] + scatterer_params["size"]) &
                    (grid_positions[1] >= scatterer_params["center_y"] - scatterer_params["size"]))
            scatterer[mask] = scatterer_params["permittivity"]

        else:
            raise ValueError("Invalid shape input")

    """ Plot scatterer """

    fig1, (ax1, ax2) = plt.subplots(ncols=2)

    original = ax1.imshow(np.real(scatterer), cmap=plt.cm.hot)  #, extent=PlotUtils.get_doi_extent())
    fig1.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
    ax1.title.set_text("Scatterer: real")

    reconstructed_real = ax2.imshow(np.imag(scatterer), cmap=plt.cm.hot)  #, extent=PlotUtils.get_doi_extent())
    fig1.colorbar(reconstructed_real, ax=ax2, fraction=0.046, pad=0.04)
    ax2.title.set_text("Scatterer: imaginary")

    plt.show()

    """" Forward problem """

    forward_solver = ForwardProblemSolver(scatterer)
    direct_field, direct_power, scattered_field, total_field, total_power = forward_solver.generate_forward_data()
    ForwardProblemSolver.get_field_plots(total_field, direct_field, scattered_field, 39)
    ForwardProblemSolver.save_data(scatterer, direct_field, direct_power, scattered_field, total_field, total_power)

    """" Inverse problem """

    model_name = "prytov_complex"
    prior = "qs2D_complex"
    params = {"alpha": 2**1}
    inverse_solver = InverseProblemSolver(direct_power, total_power, model_name, prior, params)
    real_rec, imag_rec = inverse_solver.solve()

    """ Plot reconstruction """

    fig2, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    original = ax1.imshow(scatterer, cmap=plt.cm.hot)
    fig2.colorbar(original, ax=ax1, fraction=0.046, pad=0.04)
    ax1.title.set_text("Original scatterer")

    reconstructed_real = ax2.imshow(real_rec, cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig2.colorbar(reconstructed_real, ax=ax2, fraction=0.046, pad=0.04)
    ax2.title.set_text("Real Reconstruction")

    reconstructed_imag = ax3.imshow(imag_rec, cmap=plt.cm.hot, extent=PlotUtils.get_doi_extent())
    fig2.colorbar(reconstructed_imag, ax=ax3, fraction=0.046, pad=0.04)
    ax3.title.set_text("Imaginary Reconstruction")

    plt.show()
