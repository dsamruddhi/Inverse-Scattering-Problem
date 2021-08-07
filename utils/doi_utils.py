import numpy as np

from config import Config


class DOIUtils:

    @staticmethod
    def get_grid_centroids(problem):
        """
        Returns x and y coordinates for centroids of all grids
        Two m x m arrays, one for x coordinates of the grids, one for y coordinates
        """
        if problem == "forward":
            grid_length = Config.doi["length"] / Config.doi["forward_grids"]
        elif problem == "inverse":
            grid_length = Config.doi["length"] / Config.doi["inverse_grids"]
        else:
            raise ValueError("Incorrect value of problem")

        if Config.doi["origin"] == "center":
            centroids_x = np.arange(start=- Config.doi["length"] / 2 + grid_length / 2, stop=Config.doi["length"] / 2,
                                    step=grid_length)
            centroids_y = np.arange(start=Config.doi["length"] / 2 - grid_length / 2, stop=-Config.doi["length"] / 2,
                                    step=-grid_length)
        else:
            centroids_x = np.arange(start=grid_length / 2, stop=Config.doi["length"], step=grid_length)
            centroids_y = np.arange(start=Config.doi["length"] - grid_length / 2, stop=0, step=-grid_length)
        return np.meshgrid(centroids_x, centroids_y)

    @staticmethod
    def get_grid_radius(problem):
        if problem == "forward":
            grid_length = Config.doi["length"] / Config.doi["forward_grids"]
        elif problem == "inverse":
            grid_length = Config.doi["length"] / Config.doi["inverse_grids"]
        else:
            raise ValueError("Incorrect value of problem")
        grid_radius = np.sqrt(grid_length ** 2 / np.pi)
        return grid_radius


if __name__ == '__main__':

    print(DOIUtils.get_grid_centroids())
