import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from utils.doi_utils import DOIUtils

from forward_problem.model import MethodOfMomentModel


class ForwardProblemSolver:

    transceiver = Config.sensors["transceivers"]
    rx_count = Config.sensors["count"]
    tx_count = Config.sensors["count"]
    sensor_positions = Config.sensors["positions"]
    nan_remove = True

    def __init__(self, scatterer):

        # Solver input
        self.scatterer = scatterer
        # Forward model
        self.model = MethodOfMomentModel(scatterer)

    @staticmethod
    def remove_nan_values(field):
        if ForwardProblemSolver.nan_remove:
            np.fill_diagonal(field, np.nan)
            k = field.reshape(field.size, order='F')
            l = [x for x in k if not np.isnan(x)]
            m = np.reshape(l, (ForwardProblemSolver.tx_count, ForwardProblemSolver.rx_count - 1))
            m = np.transpose(m)
            return m
        if not ForwardProblemSolver.nan_remove:
            field[np.isnan(field)] = 0
            return field

    def scatterer_independent_data(self):
        self.model.get_direct_field()
        direct_field = self.model.direct_field
        direct_field = ForwardProblemSolver.remove_nan_values(direct_field)
        direct_power = self.model.get_power_from_field(direct_field)
        return direct_field, direct_power

    def scattered_field_at_rx(self):
        self.model.find_grids_with_object()
        object_field = self.model.get_field_from_scattering()
        current = self.model.get_induced_current(object_field)
        scattered_field = self.model.get_scattered_field(current)
        scattered_field = ForwardProblemSolver.remove_nan_values(scattered_field)
        return scattered_field

    def scatterer_dependent_data(self, direct_field):
        scattered_field = self.scattered_field_at_rx()
        total_field = direct_field + scattered_field
        total_power = self.model.get_power_from_field(total_field)
        return scattered_field, total_field, total_power

    def generate_forward_data(self):
        direct_field, direct_power = self.scatterer_independent_data()
        scattered_field, total_field, total_power = self.scatterer_dependent_data(direct_field)
        return direct_field, direct_power, scattered_field, total_field, total_power

    @staticmethod
    def get_field_plots(total_field, direct_field, scattered_field, tx_num):
        plt.figure()
        plt.plot(range(ForwardProblemSolver.rx_count - 1), np.abs(total_field[:, tx_num]), label="Total Field")
        plt.plot(range(ForwardProblemSolver.rx_count - 1), np.abs(direct_field[:, tx_num]), label="Incident Field")
        plt.plot(range(ForwardProblemSolver.rx_count - 1), np.abs(scattered_field[:, tx_num]), label="Scattered Field")
        plt.axis([0, 40, 0, 0.06])
        plt.legend()
        plt.show()

    @staticmethod
    def save_data(scatterer, direct_field, direct_power, scattered_field, total_field, total_power):
        savemat('data/scatterer.mat', {"scatterer": scatterer})
        savemat('data/direct_field.mat', {"direct_field": direct_field})
        savemat('data/direct_power.mat', {"direct_power": direct_power})
        savemat('data/scattered_field.mat', {"scattered_field": scattered_field})
        savemat('data/total_field.mat', {"total_field": total_field})
        savemat('data/total_power.mat', {"total_power": total_power})


if __name__ == '__main__':

    def get_grid_permittivity(grid_positions):
        m = Config.doi["forward_grids"]
        h_side_y = 0.2
        epsilon_r = np.ones((m, m), dtype=float)
        epsilon_r[(grid_positions[0]-0.25)**2 + (grid_positions[1]-0.25)**2 <= h_side_y**2] = 2
        return epsilon_r

    grid_positions = DOIUtils.get_grid_centroids("forward")
    scatterer = get_grid_permittivity(grid_positions)
    solver = ForwardProblemSolver(scatterer)
    direct_field, direct_power, scattered_field, total_field, total_power = solver.generate_forward_data()
    ForwardProblemSolver.get_field_plots(total_field, direct_field, scattered_field, 39)
    ForwardProblemSolver.save_data(scatterer, direct_field, direct_power, scattered_field, total_field, total_power)
