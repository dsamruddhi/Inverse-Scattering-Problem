import numpy as np

from config import Config


class ShrinkagePriors:

    @staticmethod
    def identity_shrinkage(model, data, params: dict):
        m = Config.doi["inverse_grids"]
        target = np.eye(model.shape[1])
        chi = np.linalg.inv(params["intensity"]*np.transpose(model) @ model + (1 - params["intensity"])*target) @ np.transpose(model) @ data
        chi = np.reshape(chi, (m, m), order='F')
        chi = 1 + np.real(chi)
        return chi

    @staticmethod
    def sv_shrinkage(model, data, params: dict):
        m = Config.doi["inverse_grids"]
        target = np.diag(np.diag(np.transpose(model) @ model))
        chi = np.linalg.inv(params["intensity"]*np.transpose(model) @ model + (1 - params["intensity"])*target) @ np.transpose(model) @ data
        chi = np.reshape(chi, (m, m), order='F')
        chi = 1 + np.real(chi)
        return chi

    @staticmethod
    def svmc_shrinkage(model, data, params: dict):
        m = Config.doi["inverse_grids"]
        num_features = model.shape[1]
        scm = np.transpose(model) @ model
        mean_covariance = np.sum(scm * ~ np.eye(num_features, dtype=bool)) / (num_features ** 2 - num_features)
        target = np.diag(np.diag(scm)) + ~np.eye(num_features, dtype=bool) * mean_covariance
        chi = np.linalg.inv(params["intensity"]*scm + (1 - params["intensity"])*target) @ np.transpose(model) @ data
        chi = np.reshape(chi, (m, m), order='F')
        chi = 1 + np.real(chi)
        return chi
