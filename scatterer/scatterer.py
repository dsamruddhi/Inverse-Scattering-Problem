import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.doi_utils import DOIUtils


class Scatterer:

    def __init__(self, problem, inverse_type, scatterer_params: list):
        self.problem = problem
        self.inverse_type = inverse_type
        self.scatterer_params = scatterer_params
        self.grid_positions = DOIUtils.get_grid_centroids(problem)
        m = len(self.grid_positions[0])

        m = len(self.grid_positions[0])
        if self.problem == "inverse" and self.inverse_type == "ratio":
            self.scatterer = np.zeros((m, m), dtype=complex)
        else:
            self.scatterer = np.ones((m, m), dtype=complex)

    def get_permittivity(self, param):
        assert self.problem == "forward" or self.problem == "inverse"
        if self.problem == "inverse" and self.inverse_type == "ratio":
            epsilon_R = np.real(param["permittivity"])
            epsilon_I = np.imag(param["permittivity"])
            value = epsilon_I / np.sqrt(epsilon_R)
        else:
            value = param["permittivity"]
        return value

    def circle_scatterer(self, param):
        value = self.get_permittivity(param)
        self.scatterer[(self.grid_positions[0] - param["center_x"]) ** 2 + (self.grid_positions[1] - param["center_y"]) ** 2 <= param["size"] ** 2] = value

    def square_scatterer(self, param):
        value = self.get_permittivity(param)
        mask = ((self.grid_positions[0] <= param["center_x"] + param["size"]) & (self.grid_positions[0] >= param["center_x"] - param["size"]) &
                (self.grid_positions[1] <= param["center_y"] + param["size"]) & (self.grid_positions[1] >= param["center_y"] - param["size"]))
        self.scatterer[mask] = value

    def rectangle_scatterer(self, param):
        value = self.get_permittivity(param)
        mask = ((self.grid_positions[0] <= param["center_x"] + param["size1"]) & (self.grid_positions[0] >= param["center_x"] - param["size1"]) &
                (self.grid_positions[1] <= param["center_y"] + param["size2"]) & (self.grid_positions[1] >= param["center_y"] - param["size2"]))
        self.scatterer[mask] = value

    def generate(self):
        for param in self.scatterer_params:
            if param["shape"] == "circle":
                self.circle_scatterer(param)
            elif param["shape"] == "square":
                self.square_scatterer(param)
            elif param["shape"] == "rectangle":
                self.rectangle_scatterer(param)
            else:
                raise ValueError("Invalid scatterer shape")
        return self.scatterer


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    scatterer_params = [
        {
            "shape": "square",
            "center_x": 0.3,
            "center_y": 0.3,
            "size": 0.15,
            "permittivity": 3.4 + 0.25j
        },
        {
            "shape": "square",
            "center_x": -0.3,
            "center_y": 0.3,
            "size": 0.15,
            "permittivity": 77 + 7j
        }
    ]

    inv_type = None
    sc = Scatterer("forward", inv_type, scatterer_params)
    forward_scatterer = sc.generate()

    inv_type = "ratio"
    sc = Scatterer("inverse", inv_type, scatterer_params)
    inverse_scatterer = sc.generate()

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(np.real(forward_scatterer))
    plt.title("Forward scatterer: real")
    plt.subplot(1, 2, 2)
    plt.imshow(np.imag(forward_scatterer))
    plt.title("Forward scatterer: imag")
    plt.show()

    plt.figure(2)
    plt.subplot(1, 2, 1)
    plt.imshow(np.real(inverse_scatterer))
    plt.title("Inverse scatterer: real")
    plt.subplot(1, 2, 2)
    plt.imshow(np.imag(inverse_scatterer))
    plt.title("Inverse scatterer: imag")
    plt.show()
