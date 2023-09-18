# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.



from configs.linear_regression_cfg import cfg
from datasets.linear_regression_dataset import LinRegDataset
from models.linear_regression_model import LinearRegression
from utils.enums import TrainType
from utils.metrics import MSE


lin_reg_dataset = LinRegDataset(cfg)

base_functions = [lambda x: x]
base_functions2 = [lambda x, n=n: x ** n for n in range(1, 9)]
base_functions3 = [lambda x, n=n: x ** n for n in range(1, 101)]

bases = []
bases.append(base_functions)
bases.append(base_functions2)
bases.append(base_functions3)

for i in range(6):
    train_type = TrainType.gradient_descent if i < 3 else TrainType.normal_equation
    model = LinearRegression(base_functions=base_functions, train_type=train_type)
    model.train(lin_reg_dataset.inputs_train, lin_reg_dataset.targets_train)

    preds = model.calculate_model_prediction(lin_reg_dataset.inputs_train)
    train_err = MSE(preds, lin_reg_dataset.targets_train)
    preds = model.calculate_model_prediction(lin_reg_dataset.inputs_valid)
    val_err = MSE(preds, lin_reg_dataset.targets_valid)

    print("type: ", train_type, "  train: ", train_err, "  val: ", val_err)


