import numpy as np

from config import Config


class TikhonovPriors:

    @staticmethod
    def ridge(A, data, params: dict):
        m = Config.doi["inverse_grids"]
        number_of_grids = m ** 2
        if "rho" in params.keys():
            chi = params["rho"] * np.linalg.inv(params["rho"] * (A.T @ A) + (1 - params["rho"]) * np.eye(number_of_grids)) @ A.T @ data
        else:
            chi = np.linalg.inv((A.T @ A) + params["alpha"] * np.eye(number_of_grids)) @ A.T @ data
        chi = np.reshape(chi, (m, m), order='F')
        chi = 1 + np.real(chi)
        return chi

    @staticmethod
    def ridge_complex(A, data, params: dict):
        m = Config.doi["inverse_grids"]
        number_of_grids = m ** 2

        if "rho" in params.keys():
            chi = params["rho"] * np.linalg.inv(params["rho"] * (A.T @ A) + (1 - params["rho"]) * np.eye(2 * number_of_grids)) @ A.T @ data
        else:
            chi = np.linalg.inv((A.T @ A) + params["alpha"] * np.eye(2*number_of_grids)) @ A.T @ data

        chi_real = chi[:number_of_grids]
        chi_real = np.reshape(chi_real, (m, m), order='F')
        chi_real = 1 + chi_real

        chi_imag = chi[number_of_grids:]
        chi_imag = np.reshape(chi_imag, (m, m), order='F')

        return chi_real, chi_imag

    @staticmethod
    def difference_operator(m, num_grids, direction):

        d_row = np.zeros((1, num_grids))
        d_row[0, 0] = 1
        if direction == "horizontal":
            d_row[0, 1] = -1
        elif direction == "vertical":
            d_row[0, m] = -1
        else:
            raise ValueError("Invalid direction value for difference operator")

        rows = []
        rows.append(d_row)
        for i in range(0, num_grids - 1):
            shifted_row = np.roll(d_row, 1)
            shifted_row[0, 0] = 0
            rows.append(shifted_row)
            d_row = shifted_row

        d = np.vstack([row for row in rows])
        return d

    @staticmethod
    def quadratic_smoothing_2d(A, data, params: dict):
        m = Config.doi["inverse_grids"]
        number_of_grids = m ** 2

        Dx = TikhonovPriors.difference_operator(m, number_of_grids, "horizontal")
        Dy = TikhonovPriors.difference_operator(m, number_of_grids, "vertical")

        if "rho" in params.keys():
            chi = params["rho"] * np.linalg.inv(params["rho"] * (A.T @ A) + (1 - params["rho"]) * (Dx.T @ Dx + Dy.T @ Dy)) @ A.T @ data
        else:
            chi = np.linalg.inv((A.T @ A) + params["alpha"] * (Dx.T @ Dx + Dy.T @ Dy)) @ A.T @ data
        chi = np.reshape(chi, (m, m), order='F')
        chi = 1 + np.real(chi)
        return chi

    @staticmethod
    def quadratic_smoothing_2d_complex(A, data, params: dict):
        m = Config.doi["inverse_grids"]
        number_of_grids = m ** 2

        Dx = TikhonovPriors.difference_operator(m, 2*number_of_grids, "horizontal")
        Dy = TikhonovPriors.difference_operator(m, 2*number_of_grids, "vertical")

        if "rho" in params.keys():
            chi = params["rho"] * np.linalg.inv(params["rho"] * (A.T @ A) + (1 - params["rho"]) * (Dx.T @ Dx + Dy.T @ Dy)) @ A.T @ data
        else:
            chi = np.linalg.inv((A.T @ A) + params["alpha"] * (Dx.T @ Dx + Dy.T @ Dy)) @ A.T @ data
        chi_real = chi[:number_of_grids]
        chi_real = np.reshape(chi_real, (m, m), order='F')
        chi_real = 1 + chi_real

        chi_imag = chi[number_of_grids:]
        chi_imag = np.reshape(chi_imag, (m, m), order='F')
        return chi_real, chi_imag