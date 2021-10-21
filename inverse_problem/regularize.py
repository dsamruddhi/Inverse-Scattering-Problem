import numpy as np

from config import Config


class Regularizer:

    @staticmethod
    def ridge(A, data, params: dict):
        """
        Ridge regression - has an analytical solution
        Special form of Tikhonov regularization where Tikhonov matrix Q = Identity matrix
        :param A: model matrix
        :param data: form of measurement data used to solve the inverse problem
        :param params: contains "alpha" representing the regularization parameter
        :return: chi if model matrix has m**2 columns,
        chi_real. chi_imag if model matrix has 2 * m**2 columns
        """
        dim = A.shape[1]
        chi = np.linalg.inv((A.T @ A) + params["alpha"] * np.eye(dim)) @ A.T @ data

        m = Config.doi["inverse_grids"]
        if A.shape[1] == m**2:
            chi = np.reshape(chi, (m, m), order='F')
            chi = 1 + np.real(chi)
            return chi
        elif A.shape[1] == 2 * m**2:
            chi_real = chi[:m ** 2]
            chi_real = np.reshape(chi_real, (m, m), order='F')
            chi_real = 1 + chi_real
            chi_imag = chi[m ** 2:]
            chi_imag = np.reshape(chi_imag, (m, m), order='F')
            return chi_real, chi_imag
        else:
            raise ValueError("Dimensions of model matrix incorrect.")

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
    def quadratic_smoothing(A, data, params: dict):
        """
        Quadratic Smoothing - has an analytical solution
        Special form of Tikhonov regularization where Tikhonov matrix Q = Dx.T @ Dx + Dy.T @ Dy
        :param A: model matrix
        :param data: form of measurement data used to solve the inverse problem
        :param params: contains "alpha" representing the regularization parameter
        :return: chi if model matrix has m**2 columns,
        chi_real. chi_imag if model matrix has 2 * m**2 columns
        """
        m = Config.doi["inverse_grids"]
        dim = A.shape[1]

        Dx = Regularizer.difference_operator(m, dim, "horizontal")
        Dy = Regularizer.difference_operator(m, dim, "vertical")

        chi = np.linalg.inv((A.T @ A) + params["alpha"] * (Dx.T @ Dx + Dy.T @ Dy)) @ A.T @ data

        if A.shape[1] == m**2:
            chi = np.reshape(chi, (m, m), order='F')
            chi = 1 + np.real(chi)
            return chi
        elif A.shape[1] == 2 * m**2:
            chi_real = chi[:m ** 2]
            chi_real = np.reshape(chi_real, (m, m), order='F')
            chi_real = 1 + chi_real

            chi_imag = chi[m ** 2:]
            chi_imag = np.reshape(chi_imag, (m, m), order='F')
            return chi_real, chi_imag
        else:
            raise ValueError("Dimensions of model matrix incorrect.")
