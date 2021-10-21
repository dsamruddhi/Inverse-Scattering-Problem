import os
import sys
import numpy as np
from scipy.special import jv as bessel1
from scipy.special import hankel1

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config
from utils.doi_utils import DOIUtils


class LinearInverse:

    def __init__(self):

        # System parameters
        self.frequency = Config.system["frequency"]
        self.wavelength = 3e8 / self.frequency
        self.wave_number = 2*np.pi / self.wavelength
        self.impedance = 120*np.pi

        # Room parameters
        self.m = Config.doi["inverse_grids"]
        self.number_of_grids = self.m ** 2
        self.grid_positions = DOIUtils.get_grid_centroids("inverse")
        self.grid_radius = DOIUtils.get_grid_radius("inverse")

        # Sensor parameters
        self.transceiver = Config.sensors["transceivers"]
        self.number_of_rx = Config.sensors["count"]
        self.number_of_tx = Config.sensors["count"]
        self.sensor_positions = Config.sensors["positions"]

        self.nan_remove = True
        self.noise_level = 0

    def get_direct_field(self):
        """
        Field from transmitter to receiver
        Output dimension - number of transmitters x number of receivers
        """
        receiver_x = [pos[0] for pos in self.sensor_positions]
        receiver_y = [pos[1] for pos in self.sensor_positions]

        transmitter_x = [pos[0] for pos in self.sensor_positions]
        transmitter_y = [pos[1] for pos in self.sensor_positions]

        [xtd, xrd] = np.meshgrid(transmitter_x, receiver_x)
        [ytd, yrd] = np.meshgrid(transmitter_y, receiver_y)
        dist = np.sqrt((xtd - xrd) ** 2 + (ytd - yrd) ** 2)
        direct_field = (1j / 4) * hankel1(0, self.wave_number * dist)
        return direct_field

    def get_incident_field(self):
        """
        Field from transmitter on every incident grid
        Output dimension - number of transmitters x number of grids
        """
        transmitter_x = [pos[0] for pos in self.sensor_positions]
        transmitter_y = [pos[1] for pos in self.sensor_positions]

        grid_x = self.grid_positions[0]
        grid_x = grid_x.reshape(grid_x.size, order='F')

        grid_y = self.grid_positions[1]
        grid_y = grid_y.reshape(grid_y.size, order='F')

        [xti, xsi] = np.meshgrid(transmitter_x, grid_x)
        [yti, ysi] = np.meshgrid(transmitter_y, grid_y)

        dist = np.sqrt((xti - xsi)**2 + (yti - ysi)**2)
        incident_field = (1j/4) * hankel1(0, self.wave_number * dist)
        return incident_field

    def get_greens_integral(self):
        transmitter_x = [pos[0] for pos in self.sensor_positions]
        transmitter_y = [pos[1] for pos in self.sensor_positions]

        grid_x = self.grid_positions[0]
        grid_x = grid_x.reshape(grid_x.size, order='F')

        grid_y = self.grid_positions[1]
        grid_y = grid_y.reshape(grid_y.size, order='F')

        [xtg, xsg] = np.meshgrid(transmitter_x, grid_x)
        [ytg, ysg] = np.meshgrid(transmitter_y, grid_y)

        dist = np.sqrt((xtg - xsg)**2 + (ytg - ysg)**2)
        integral = (1j * np.pi * self.grid_radius / (2 * self.wave_number)) * \
            bessel1(1, self.wave_number * self.grid_radius) * hankel1(0, self.wave_number * np.transpose(dist))
        return integral
