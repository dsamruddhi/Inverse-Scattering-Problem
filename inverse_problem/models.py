""" Class inheritance = Model -> PRytov -> PRytovComplex """
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import Config


class Model:

    def __init__(self):

        # Physical parameters
        self.frequency = Config.system["frequency"]
        self.wavelength = 3e8 / self.frequency
        self.wave_number = 2 * np.pi / self.wavelength

        # Room parameters
        self.m = Config.doi["inverse_grids"]
        self.number_of_grids = self.m ** 2

        # Sensor parameters
        self.transceiver = Config.sensors["transceivers"]
        self.number_of_rx = Config.sensors["count"]
        self.number_of_tx = Config.sensors["count"]
        self.sensor_positions = Config.sensors["positions"]
        self.sensor_links = Config.sensors["links"]

        # Other parameters
        self.nan_remove = True

    def get_model(self, *args, **kwargs):
        return

    def get_data(self, *args, **kwargs):
        return


class PRytov(Model):

    def get_model(self, direct_field, incident_field, integral_values):
        A = np.zeros((len(self.sensor_links), self.number_of_grids), dtype=complex)
        for i, pair in enumerate(self.sensor_links):
            A[i, :] = np.real(self.wave_number ** 2
                              * np.divide(np.multiply(integral_values[pair[1], :],
                                                      np.transpose(incident_field[:, pair[0]])),
                                          direct_field[pair[1], pair[0]]))
        return A

    @staticmethod
    def get_data(total_power, direct_power):
        data = (total_power - direct_power) / (10 * np.log10(np.exp(2)))
        data = data.reshape(data.size, order='F')
        return data


class PRytovComplex(PRytov):

    def get_model(self, direct_field, incident_field, integral_values):
        A = np.zeros((len(self.sensor_links), self.number_of_grids), dtype=complex)
        for i, pair in enumerate(self.sensor_links):
            A[i, :] = (self.wave_number ** 2) * np.divide(np.multiply(integral_values[pair[1], :],
                                                                      np.transpose(incident_field[:, pair[0]])),
                                                          direct_field[pair[1], pair[0]])

        A_real = np.real(A)
        A_imag = np.imag(A)
        A_final = np.concatenate((A_real, -A_imag), axis=1)
        return A_final


class PRytovImag(PRytov):

    def get_model(self, direct_field, incident_field, integral_values):
        A = np.zeros((len(self.sensor_links), self.number_of_grids), dtype=complex)
        for i, pair in enumerate(self.sensor_links):
            A[i, :] = (self.wave_number ** 2) * np.divide(np.multiply(integral_values[pair[1], :],
                                                                      np.transpose(incident_field[:, pair[0]])),
                                                          direct_field[pair[1], pair[0]])

        A_imag = np.imag(A)
        return -A_imag
