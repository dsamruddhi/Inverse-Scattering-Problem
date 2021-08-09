import numpy as np
from scipy.special import jv as bessel1
from scipy.special import hankel1
from datetime import datetime

from config import Config
from utils.doi_utils import DOIUtils


class MethodOfMomentModel:

    # System parameters
    frequency = Config.system["frequency"]
    wavelength = 3e8 / frequency
    wave_number = 2 * np.pi / wavelength
    impedance = 120 * np.pi

    # Room parameters
    m = Config.doi["forward_grids"]
    num_grids = m ** 2
    grid_positions = DOIUtils.get_grid_centroids("forward")
    grid_radius = DOIUtils.get_grid_radius("forward")

    # Sensor parameters
    transceiver = Config.sensors["transceivers"]
    rx_count = Config.sensors["count"]
    tx_count = Config.sensors["count"]
    sensor_positions = Config.sensors["positions"]

    nan_remove = True
    noise_level = 0

    # Constants used in code
    C1 = -impedance * np.pi * (grid_radius / 2)
    C2 = bessel1(1, wave_number * grid_radius)
    C3 = hankel1(1, wave_number * grid_radius)

    def __init__(self, grid_permittivities):

        self.grid_permittivities = grid_permittivities
        self.grid_object_indices = None

        # Do not change with scatterer
        self.direct_field = None  # Direct field at receiver
        self.direct_power = None  # Direct power at receiver

        self.incident_field = None  # Incident field on every grid of DOI

        # Change with scatterer
        self.scattered_field = None  # Field scattered due to object obtained at the receiver

        self.total_field = None  # Total field obtained at receiver
        self.total_power = None  # Total power obtained at receiver

    def get_direct_field(self):
        """
        Field from transmitter to receiver
        Output dimension - number of transmitters x number of receivers
        Does not change with scatterer
        """
        receiver_x = [pos[0] for pos in MethodOfMomentModel.sensor_positions]
        receiver_y = [pos[1] for pos in MethodOfMomentModel.sensor_positions]
        transmitter_x = [pos[0] for pos in MethodOfMomentModel.sensor_positions]
        transmitter_y = [pos[1] for pos in MethodOfMomentModel.sensor_positions]

        [xtd, xrd] = np.meshgrid(transmitter_x, receiver_x)
        [ytd, yrd] = np.meshgrid(transmitter_y, receiver_y)
        dist = np.sqrt((xtd - xrd)**2 + (ytd - yrd)**2)
        self.direct_field = (1j/4) * hankel1(0, MethodOfMomentModel.wave_number * dist)

    def get_incident_field(self):
        """
        Field from transmitter on every incident grid
        Output dimension - number of transmitters x number of grids
        Does not change with scatterer
        """
        transmitter_x = [pos[0] for pos in MethodOfMomentModel.sensor_positions]
        transmitter_y = [pos[1] for pos in MethodOfMomentModel.sensor_positions]

        grid_x = MethodOfMomentModel.grid_positions[0]
        grid_x = grid_x.reshape(grid_x.size, order='F')

        grid_y = MethodOfMomentModel.grid_positions[1]
        grid_y = grid_y.reshape(grid_y.size, order='F')

        [xti, xsi] = np.meshgrid(transmitter_x, grid_x)
        [yti, ysi] = np.meshgrid(transmitter_y, grid_y)

        dist = np.sqrt((xti - xsi)**2 + (yti - ysi)**2)
        self.incident_field = (1j/4) * hankel1(0, MethodOfMomentModel.wave_number * dist)

    def find_grids_with_object(self):
        # Unroll all grid numbers into one array, return the grid numbers that contain objects
        self.unrolled_permittivities = self.grid_permittivities.reshape(self.grid_permittivities.size, order='F')
        self.object_grid_indices = np.nonzero(self.unrolled_permittivities != 1)
        self.object_grid_indices = self.object_grid_indices[0]

    def get_field_from_scattering(self):
        """ Object field is a 2D array that captures the field on every point scatterer
        due to every other point scatterer """

        Z = np.zeros((len(self.object_grid_indices), len(self.object_grid_indices)), dtype=np.complex64)
        unroll_x = MethodOfMomentModel.grid_positions[0].reshape(MethodOfMomentModel.grid_positions[0].size, order='F')
        unroll_y = MethodOfMomentModel.grid_positions[1].reshape(MethodOfMomentModel.grid_positions[1].size, order='F')
        x_obj = unroll_x[self.object_grid_indices]
        y_obj = unroll_y[self.object_grid_indices]

        for index, value in enumerate(self.object_grid_indices):
            x_incident = x_obj[index]
            y_incident = y_obj[index]

            dipole_distances = np.sqrt((x_incident - x_obj) ** 2 + (y_incident - y_obj) ** 2)

            a1 = hankel1(0, MethodOfMomentModel.wave_number * dipole_distances)
            b1 = MethodOfMomentModel.impedance * self.unrolled_permittivities[value] / (MethodOfMomentModel.wave_number * (self.unrolled_permittivities[value] - 1))

            z1 = MethodOfMomentModel.C1 * MethodOfMomentModel.C2 * a1
            z1[index] = MethodOfMomentModel.C1 * MethodOfMomentModel.C3 - 1j * b1

            assert len(z1) == len(dipole_distances)
            Z[index, :] = z1

        return Z

    def get_induced_current(self, object_field):
        # Only consider that part of incident field which falls on grids containing object
        self.get_incident_field()
        incident_field_on_object = - self.incident_field[self.object_grid_indices]

        J1 = np.linalg.inv(object_field) @ incident_field_on_object

        current = np.zeros((MethodOfMomentModel.m**2, MethodOfMomentModel.tx_count), dtype=complex)
        for i in range(len(self.object_grid_indices)):
            current[self.object_grid_indices[i], :] = J1[i, :]

        return current

    @staticmethod
    def get_scattered_field(current):

        transmitter_x = [pos[0] for pos in MethodOfMomentModel.sensor_positions]
        transmitter_y = [pos[1] for pos in MethodOfMomentModel.sensor_positions]

        grid_x = MethodOfMomentModel.grid_positions[0]
        grid_x = grid_x.reshape(grid_x.size, order='F')

        grid_y = MethodOfMomentModel.grid_positions[1]
        grid_y = grid_y.reshape(grid_y.size, order='F')

        [xts, xss] = np.meshgrid(transmitter_x, grid_x)
        [yts, yss] = np.meshgrid(transmitter_y, grid_y)

        dist = np.sqrt((xts - xss)**2 + (yts - yss)**2)
        ZZ = - MethodOfMomentModel.impedance * np.pi * (MethodOfMomentModel.grid_radius/2) * \
             bessel1(1, MethodOfMomentModel.wave_number * MethodOfMomentModel.grid_radius) * \
             hankel1(0, MethodOfMomentModel.wave_number * np.transpose(dist))
        scattered_field = ZZ @ current

        return scattered_field

    def remove_nan_values(self, field):
        if self.nan_remove:
            np.fill_diagonal(field, np.nan)
            k = field.reshape(field.size, order='F')
            l = [x for x in k if not np.isnan(x)]
            m = np.reshape(l, (self.tx_count, self.rx_count - 1))
            m = np.transpose(m)
            return m
        if not self.nan_remove:
            field[np.isnan(field)] = 0
            return field

    def transceiver_manipulation(self):
        if self.transceiver:
            self.direct_field = self.remove_nan_values(self.direct_field)
            self.scattered_field = self.remove_nan_values(self.scattered_field)
            self.total_field = self.remove_nan_values(self.total_field)

    @staticmethod
    def get_power_from_field(field):
        power = (np.abs(field)**2) * (MethodOfMomentModel.wavelength**2) / (4*np.pi*MethodOfMomentModel.impedance)
        power = 10 * np.log10(power / 1e-3)
        return power
