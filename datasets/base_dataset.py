import numpy as np
from abc import ABC, abstractmethod


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

    def get_inputs_shape(self):
        return self.inputs.shape

    def get_targets_shape(self):
        return self.targets.shape

    def _divide_into_sets(self):
        # TODO define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test
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
