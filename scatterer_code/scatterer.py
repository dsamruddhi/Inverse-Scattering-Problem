import numpy as np

from utils.doi_utils import DOIUtils


class Scatterer:

    def __init__(self, problem, scatterer_params: list):
        self.problem = problem
        self.scatterer_params = scatterer_params
        self.grid_positions = DOIUtils.get_grid_centroids(problem)
        m = len(self.grid_positions[0])
        self.scatterer = np.ones((m, m), dtype=float)

    def circle_scatterer(self, param):
        self.scatterer[(self.grid_positions[0] - param["center_x"]) ** 2 + (self.grid_positions[1] - param["center_y"]) ** 2 <= param["size"] ** 2] = param["permittivity"]

    def square_scatterer(self, param):
        mask = ((self.grid_positions[0] <= param["center_x"] + param["size"]) & (self.grid_positions[0] >= param["center_x"] - param["size"]) &
                (self.grid_positions[1] <= param["center_y"] + param["size"]) & (self.grid_positions[1] >= param["center_y"] - param["size"]))
        self.scatterer[mask] = param["permittivity"]

    def generate(self):
        for param in self.scatterer_params:
            if param["shape"] == "circle":
                self.circle_scatterer(param)
            elif param["shape"] == "square":
                self.square_scatterer(param)
            else:
                raise ValueError("Invalid scatterer shape")
        return self.scatterer
