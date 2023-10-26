from abc import ABC, abstractmethod

import numpy as np


class BaseDataset(ABC):

    def __init__(self, train_set_percent, valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass

    @property
    @abstractmethod
    def d(self):
        # inputs variables
        pass

    def get_inputs_shape(self):
        return self.inputs.shape

    def get_targets_shape(self):
        return self.targets.shape

    def divide_into_sets(self):
        # TODO define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test; you can use your code from previous homework

        n = self.get_targets_shape()[0]
        indexes = np.arange(n)
        np.random.shuffle(indexes)
        self.inputs_train = self.inputs[indexes[:int(self.train_set_percent * n)]]
        self.targets_train = self.targets[indexes[:int(self.train_set_percent * n)]]
        self.inputs_valid = self.inputs[
            indexes[int(self.train_set_percent * n): int((self.train_set_percent + self.valid_set_percent) * n)]]
        self.targets_valid = self.targets[
            indexes[int(self.train_set_percent * n): int((self.train_set_percent + self.valid_set_percent) * n)]]
        self.inputs_test = self.inputs[indexes[int((self.train_set_percent + self.valid_set_percent) * n):]]
        self.targets_test = self.targets[indexes[int((self.train_set_percent + self.valid_set_percent) * n):]]

    def normalization(self):
        # TODO write normalization method BONUS TASK
        pass

    def get_data_stats(self):
        # TODO calculate mean and std of inputs vectors of training set by each dimension
        std = np.std(self.inputs_train, axis=0)
        mean = np.mean(self.inputs_train, axis=0)
        return mean, std

    def standartization(self):
        # TODO write standardization method (use stats from __get_data_stats)
        #   DON'T USE LOOP
        std, mean = self.get_data_stats()



class BaseClassificationDataset(BaseDataset):

    @property
    @abstractmethod
    def k(self):
        # number of classes
        pass

    @staticmethod
    def onehotencoding(targets: np.ndarray, number_classes: int) -> np.ndarray:
        # TODO create matrix of onehot encoding vactors for input targets
        # it is possible to do it without loop try make it without loop
        pass