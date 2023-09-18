import os

from easydict import EasyDict
from utils.enums import TrainType

cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = os.path.join('dataframes', 'linear_regression_dataset.csv')

# cfg.base_functions contains callable functions to transform input features.
# E.g., for polynomial regression: [lambda x: x, lambda x: x**2]
# TODO You should populate this list with suitable functions based on the requirements.

#cfg.base_functions = [lambda x: x, lambda x: x ** 2, lambda x: x**3, lambda x: x**4, lambda x: x**5]
cfg.base_functions = [lambda x, n=n: x ** n for n in range(10)]

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

# Specifies the type of training algorithm to be used
cfg.train_type = TrainType.gradient_descent

# how many times the algorithm will process the entire dataset for gradient descent algorithm
cfg.epoch = 100



