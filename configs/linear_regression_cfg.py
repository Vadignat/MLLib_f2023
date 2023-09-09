import os

from easydict import EasyDict

cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = os.path.join('..', 'dataframes', 'linear_regression_dataset.csv')

cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1





