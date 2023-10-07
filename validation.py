# TODO:
#  1) Load the dataset using pandas read_csv function.
#  2) Split the dataset into training, validation, and test sets.
#  Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  Use class from datasets.linear_regression_dataset.py
#  3) Define hyperparameters space
#  4) Use loop where you randomly choose hypeparameter from space and train model
#  5) Create experiment name using code from logging_example.py
#  6) Initialize the Linear Regression model using the provided `LinearRegression` class
#  7) Log hyperparameters to neptune
#  8) Train the model using the training data and gradient descent,
#  log MSE and cost function on validation and trainig sets
#  9) Log final mse on validation set after trainig
#  10) Save model if it is showing best mse on validation set
import pickle

import numpy as np
from logs.Logger import Logger
from configs.linear_regression_cfg import cfg
from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression
from utils.common_functions import generate_experiment_name
from utils.enums import TrainType
from utils.metrics import MSE


lin_reg_dataset = LinRegDataset(cfg, inputs_cols=['x_0', 'x_1', 'x_2'])

n = lin_reg_dataset.get_inputs_shape()[1]
base_functions = []
for i in range(n):
    base_functions.extend([lambda x, i=i, n=n: x[i] ** n for n in range(1, 11)])
    base_functions.extend([lambda x, i=i: np.sin(x[i]), lambda x, i=i: np.cos(x[i]),
                           lambda x, i=i: np.exp(x[i]), lambda x, i=i: np.log(x[i])])

M_min = 1
M_max = len(base_functions)

rc_min = 0.
rc_max = 1

lr_min = 0.01
lr_max = 0.2


for i in range(30):
    M = np.random.randint(M_min, M_max + 1)
    rc = np.random.uniform(rc_min, rc_max)
    lr = np.random.uniform(lr_min, lr_max)
    M_base_functions = np.random.choice(base_functions, M)

    experiment_name, base_function_str = generate_experiment_name(M_base_functions, rc, lr)
    logger = Logger(env_path=cfg.env_path, project=cfg.project_name, experiment_name=experiment_name)

    logger.log_hyperparameters(params={
        'base_function': base_function_str,
        'regularisation_coefficient': rc,
        'learning_rate': lr
    })

    model = LinearRegression(
        base_functions=M_base_functions,
        train_type=TrainType.gradient_descent,
        learning_rate=lr,
        reg_coefficient=rc,
        experiment_name=experiment_name
    )
    model.train(lin_reg_dataset.inputs_train, lin_reg_dataset.targets_train)

    preds = model.calculate_model_prediction(lin_reg_dataset.inputs_valid)
    val_err = MSE(preds, lin_reg_dataset.targets_valid)
    logger.log_final_val_mse(val_err)
    logger.run.stop()

    if i == 0 or best_err > val_err:
        best_err = val_err
        best_model = model


print("best mse:", best_err)
best_model.save('saved_models/linear-regression.pkl')


'''
model = LinearRegression().load('saved_models/linear-regression.pkl')
print(model.neptune_logger)
'''

