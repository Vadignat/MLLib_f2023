# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.
import sys

import numpy as np


def plan_matrix(base_functions, inputs: np.ndarray) -> np.ndarray:
    res = np.ones_like(inputs)
    for func in base_functions:
        res = np.append(res, func(inputs), axis=1)
    return res


base_functions = [lambda x: x, lambda x: x ** 2, lambda x: x**3, lambda x: x**4, lambda x: x**5]
a = np.eye(10, 3)



