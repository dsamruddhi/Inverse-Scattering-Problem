import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from config import Config


class SparsityPriors:

    @staticmethod
    def lasso(model, data, params: dict):
        # alpha=1e-6, positive=True
        ls = Lasso(params["alpha"], positive=True)
        model = np.real(model)
        data = np.real(data)
        ls.fit(model, data)
        chi = ls.coef_
        m = Config.doi["inverse_grids"]
        chi = np.reshape(chi, (m, m), order='F')
        chi = 1 + np.real(chi)
        return chi

    @staticmethod
    def elastic_net(model, data, params: dict):
        # alpha=1e-5, l1_ratio=0.7, positive=True
        ls = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], positive=params["positive"])
        model = np.real(model)
        data = np.real(data)
        ls.fit(model, data)
        chi = ls.coef_
        m = Config.doi["inverse_grids"]
        chi = np.reshape(chi, (m, m), order='F')
        chi = 1 + np.real(chi)
        return chi
