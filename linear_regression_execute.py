# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.
import numpy as np

from configs.linear_regression_cfg import cfg
from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression
from utils.enums import TrainType
from utils.metrics import MSE
from utils.visualisation import Visualisation

lin_reg_dataset = LinRegDataset(cfg)

base_functions = [lambda x: x]
base_functions2 = [lambda x, n=n: x ** n for n in range(1, 9)]
base_functions3 = [lambda x, n=n: x ** n for n in range(1, 101)]

bases = [base_functions, base_functions2, base_functions3]

y_values_list = []

x = lin_reg_dataset.inputs_valid
y = lin_reg_dataset.targets_valid
sorted_indices = np.argsort(x)
x = x[sorted_indices]
y = y[sorted_indices]

errors = []
best_err = 10000
for i in range(6):
    train_type = TrainType.gradient_descent if i % 2 == 0 else TrainType.normal_equation
    model = LinearRegression(base_functions=bases[i//2], train_type=train_type, learning_rate=0.1)
    model.train(lin_reg_dataset.inputs_train, lin_reg_dataset.targets_train)

    preds = model.calculate_model_prediction(lin_reg_dataset.inputs_train)
    train_err = MSE(preds, lin_reg_dataset.targets_train)
    preds = model.calculate_model_prediction(x)
    val_err = MSE(preds, y)
    errors.append(val_err)

    if best_err > val_err:
        best_err = val_err
        best_model = model
    y_values_list.append(preds)
    print("type: ", train_type.name, "  train: ", train_err, "  val: ", val_err)

preds = best_model.calculate_model_prediction(lin_reg_dataset.inputs_test)
test_err = MSE(preds, lin_reg_dataset.targets_test)
print()
print("Значение MSE лучшей модели:", test_err)

names = ["Gradient Descent", "Normal equation"]
vis = Visualisation()
fig1 = vis.compare_model_predictions(x, y_values_list[:2], y, "Максимальная степень полинома 1, "
                                                              f"MSE{errors[0], errors[1]}", names=names)
fig2 = vis.compare_model_predictions(x, y_values_list[2:4], y, "Максимальная степень полинома 8, "
                                                               f"MSE{errors[2], errors[3]}", names=names)
fig3 = vis.compare_model_predictions(x, y_values_list[4:], y, "Максимальная степень полинома 100, "
                                                              f"MSE{errors[4], errors[5]}", names=names)

fig1.show()
fig2.show()
fig3.show()



