import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inverse_problem.inverse import LinearInverse
from inverse_problem.models import PRytov, PRytovComplex, PRytovImag
from inverse_problem.regularize import Regularizer


class InverseProblemSolver:

    def __init__(self, direct_power, total_power, model_name, prior, params):
        self.direct_power = direct_power
        self.total_power = total_power

        self.model_name = model_name
        self.prior = prior
        self.params = params

    def get_model_class(self):
        if self.model_name == "prytov":
            model_class = PRytov()
        elif self.model_name == "prytov_complex":
            model_class = PRytovComplex()
        elif self.model_name == "prytov_imag":
            model_class = PRytovImag()
        else:
            raise ValueError("Incorrect model name, input should be either 'prytov' or 'prytov_complex'")
        return model_class

    def get_inverse_model(self):
        inverse_problem = LinearInverse()
        direct_field = inverse_problem.get_direct_field()
        incident_field = inverse_problem.get_incident_field()
        integral_values = inverse_problem.get_greens_integral()

        model_class = self.get_model_class()
        A = model_class.get_model(direct_field, incident_field, integral_values)
        return A

    def get_measurement_data(self):
        model_class = self.get_model_class()
        y = model_class.get_data(self.total_power, self.direct_power)
        return y

    def get_regularizer(self):
        mapping = {
            "ridge": Regularizer.ridge,
            "qs2D": Regularizer.quadratic_smoothing,
            "lasso": Regularizer.lasso,
            "elasticnet": Regularizer.elasticnet
        }
        if self.prior not in mapping.keys():
            raise ValueError("Invalid prior value")
        return mapping[self.prior]

    def solve(self):
        model = self.get_inverse_model()
        data = self.get_measurement_data()
        regularizer = self.get_regularizer()
        chi = regularizer(model, data, self.params)
        return chi
