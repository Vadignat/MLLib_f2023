from typing import Union

import cloudpickle
import numpy as np
from easydict import EasyDict

from logs.Logger import Logger
from utils.metrics import accuracy, confusion_matrix


class LogReg:

    def __init__(self, cfg: EasyDict, number_classes: int, input_vector_dimension: int, experiment_name: str = None):
        self.k = number_classes
        self.d = input_vector_dimension
        self.cfg = cfg
        self.neptune_logger = Logger(cfg.env_path, cfg.project_name, experiment_name)
        #getattr(self, f'weights_init_{cfg.weights_init_type.name}')(**cfg.weights_init_kwargs)
        getattr(self, f'weights_init_{cfg.weights_init_type.name}')()

    def weights_init_xavier_normal(self):
        # TODO init weights with Xavier normal W ~ N(0, sqrt(2 / (D + K)))
        self.W = np.random.normal(0, np.sqrt(2 / (self.d + self.k)), (self.k, self.d))
        self.b = np.random.normal(0, np.sqrt(2 / (self.d + self.k)), (self.k, ))

    def weights_init_xavier_uniform(self):
        # init weights with Xavier uniform W ~ U(-sqrt(6 / (D + K)), sqrt(6 / (D + K)))
        self.W = np.random.uniform(-np.sqrt(6 / (self.d + self.k)), np.sqrt(6 / (self.d + self.k)),
                                   (self.k, self.d))
        self.b = np.random.uniform(-np.sqrt(6 / (self.d + self.k)), np.sqrt(6 / (self.d + self.k)), self.k)

    def weights_init_he_normal(self):
        # init weights with He normal W ~ N(0, sqrt(2 / D))
        self.W = np.random.normal(0, np.sqrt(2 / self.d), (self.k, self.d))
        self.b = np.random.normal(0, np.sqrt(2 / self.d), self.k)

    def weights_init_he_uniform(self):
        #  init weights with He uniform W ~ U(-sqrt(6 / D), sqrt(6 / D)
        self.W = np.random.uniform(-np.sqrt(6 / self.d), np.sqrt(6 / self.d), (self.k, self.d))
        self.b = np.random.uniform(-np.sqrt(6 / self.d), np.sqrt(6 / self.d), self.k)

    def __softmax(self, model_output: np.ndarray) -> np.ndarray:
        """
               Computes the softmax function on the model output.

               The formula for softmax function is:
               y_j = e^(z_j) / Σ(i=0 to K-1) e^(z_i)

               where:
               - y_j is the softmax probability of class j,
               - z_j is the model output for class j before softmax,
               - K is the total number of classes,
               - Σ denotes summation.

               For numerical stability, subtract the max value of model_output before exponentiation:
               z_j = z_j - max(model_output)

               Parameters:
               model_output (np.ndarray): The model output before softmax.

               Returns:
               np.ndarray: The softmax probabilities.
            TODO implement this function
        """
        z = model_output - np.max(model_output)
        z = np.exp(z)
        sum = np.sum(z)
        return z / sum

    def get_model_confidence(self, inputs: np.ndarray) -> np.ndarray:
        """
                Calculates model confidence using the formula:
                y(x, b, W) = Softmax(Wx + b) = Softmax(z)

                Parameters:
                inputs (np.ndarray): The input data.

                Returns:
                np.ndarray: The model confidence.
        """
        z = self.__get_model_output(inputs)
        y = self.__softmax(z)
        return y

    def __get_model_output(self, inputs: np.ndarray) -> np.ndarray:
        """
        This function computes the model output by applying a linear transformation
        to the input data.

        The linear transformation is defined by the equation:
        z = W * x + b

        where:
        - W (a KxD matrix) represents the weight matrix,
        - x (a DxN matrix, also known as 'inputs') represents the input data,
        - b (a vector of length K) represents the bias vector,
        - z represents the model output before activation.

        Returns:
        np.ndarray: The model output before softmax.

        TODO implement this function  using matrix multiplication DO NOT USE LOOPS
        """
        '''
        if inputs.ndim == 1:
            return self.W @ inputs + self.b

        z = []
        for i in range(inputs.shape[0]):
            z += self.W @ inputs[i] + self.b
        return z
        '''
        if inputs.ndim == 1:
            return self.W @ inputs + self.b

        z = self.W @ inputs.T + self.b[:, np.newaxis]
        return z.T

    def __get_gradient_w(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        # calculate gradient for w
        y = model_confidence - targets
        return np.outer(y, inputs)

    def __get_gradient_b(self, targets: np.ndarray, model_confidence: np.ndarray) -> np.ndarray:
        # calculate gradient for b
        y = model_confidence - targets
        return y

    def __weights_update(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: np.ndarray):
        #  update model weights
        self.W -= self.cfg.gamma * self.__get_gradient_w(inputs, targets, model_confidence)
        self.b -= self.cfg.gamma * self.__get_gradient_b(targets, model_confidence)

    def __gradient_descent_step(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                epoch: int, inputs_valid: Union[np.ndarray, None] = None,
                                targets_valid: Union[np.ndarray, None] = None):
        #  one step in Gradient descent:
        #  calculate model confidence;
        #  target function value calculation;
        #
        #  update weights
        #   you can add some other steps if you need
        # log your results in Neptune
        """
        :param targets_train: onehot-encoding
        :param epoch: number of loop iteration
        """
        for i in range(inputs_train.shape[0]):
            z = self.get_model_confidence(inputs_train[i])
            loss_train = self.__target_function_value(inputs_train, targets_train)
            print(f"Train target function value {epoch}: ", loss_train)
            train_accuracy, train_matrix = self.__validate(inputs_train, targets_train)
            print(f"Train accuracy {epoch}: ", train_accuracy)
            print(f" Train confusion matrix {epoch}: ")
            print(train_matrix)
            self.neptune_logger.save_param(
                'train',
                ['target_function_value', 'accuracy'],
                [loss_train, train_accuracy]
            )

            loss_val = self.__target_function_value(inputs_valid, targets_valid)
            val_accuracy, val_matrix = self.__validate(inputs_valid, targets_valid)
            print(f"Val accuracy {epoch}: ", val_accuracy)
            print(f"Val confusion matrix {epoch}: ")
            print(val_matrix)
            if epoch % 5 == 0:
                self.neptune_logger.save_param(
                    'val',
                    ['target_function_value', 'accuracy'],
                    [loss_val, val_accuracy]
                )

            self.__weights_update(inputs_train[i], targets_train[i], z)

    def gradient_descent_epoch(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                               inputs_valid: Union[np.ndarray, None] = None,
                               targets_valid: Union[np.ndarray, None] = None):
        # loop stopping criteria - number of iterations of gradient_descent
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        for epoch in range(self.cfg.nb_epoch):
            self.__gradient_descent_step(inputs_train, targets_train, epoch, inputs_valid, targets_valid)

    def gradient_descent_gradient_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                       inputs_valid: Union[np.ndarray, None] = None,
                                       targets_valid: Union[np.ndarray, None] = None):
        #  gradient_descent with gradient norm stopping criteria BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def gradient_descent_difference_norm(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                         inputs_valid: Union[np.ndarray, None] = None,
                                         targets_valid: Union[np.ndarray, None] = None):
        #  gradient_descent with stopping criteria - norm of difference between ￼w_k-1 and w_k;￼BONUS TASK
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def gradient_descent_metric_value(self, inputs_train: np.ndarray, targets_train: np.ndarray,
                                      inputs_valid: Union[np.ndarray, None] = None,
                                      targets_valid: Union[np.ndarray, None] = None):
        #  gradient_descent with stopping criteria - metric (accuracy, f1 score or other) value on validation set is not growing;￼
        # while not stopping criteria
        #   self.__gradient_descent_step(inputs, targets)
        pass

    def train(self, inputs_train: np.ndarray, targets_train: np.ndarray,
              inputs_valid: Union[np.ndarray, None] = None, targets_valid: Union[np.ndarray, None] = None):
        getattr(self, f'gradient_descent_{self.cfg.gd_stopping_criteria.name}')(inputs_train, targets_train,
                                                                                inputs_valid,
                                                                                targets_valid)

    def __target_function_value(self, inputs: np.ndarray, targets: np.ndarray,
                                z: Union[np.ndarray, None] = None) -> float:
        """
        This function computes the target function value based on the formula:

        Σ(i=0 to N-1) Σ(k=0 to K-1) t_ik * (ln(Σ(l=0 to K-1) e^(z_il)) - z_ik)
        where:
        - N is the size of the data set,
        - K is the number of classes,
        - t_{ik} is the target value for data point i and class k,
        - z_{il} is the model output before softmax for data point i and class l,
        - z is the model output before softmax (matrix z).

        Parameters:
        inputs (np.ndarray): The input data.
        targets (np.ndarray): The target data.
        z (Union[np.ndarray, None]): The model output before softmax. If None, it will be computed.

        Returns:
        float: The value of the target function.
        TODO implement this function
        """

        if z is None:
            z = self.__get_model_output(inputs)

        return np.sum(targets * (np.log(np.sum(np.exp(z), axis=1))[:, np.newaxis]) - z)



    def __validate(self, inputs: np.ndarray, targets: np.ndarray, model_confidence: Union[np.ndarray, None] = None):
        #  metrics calculation: accuracy, confusion matrix
        if model_confidence is None:
            model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=1)
        acry = accuracy(predictions, targets)
        matrix = confusion_matrix(predictions, targets, num_classes=self.k)
        return acry, matrix

    def __call__(self, inputs: np.ndarray):
        model_confidence = self.get_model_confidence(inputs)
        predictions = np.argmax(model_confidence, axis=0)
        return predictions

    def __getstate__(self):
        state = self.__dict__.copy()
        # Исключаем self.neptune_logger из словаря состояния
        state['neptune_logger'] = None
        return state

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, filepath, cfg):
        with open(filepath, 'rb') as f:
            model = cloudpickle.load(f)
            model.neptune_logger = Logger(cfg.env_path, cfg.project_name, model.experiment_name)
            return model
