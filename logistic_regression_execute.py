#TODO
# инициализация класса набора данных, стандартизация данных, разделение на выборки, построение onehot encoding вектора
# инициализация класса логистической регрессии
# обучение модели, логирование в Нептун
# сохранение модели

import numpy as np
from configs.logistic_regression_cfg import cfg
from datasets.digits_dataset import Digits
from models.logistic_regression_model import LogReg
from utils.metrics import accuracy, confusion_matrix

log_reg_dataset = Digits(cfg)

model = LogReg(cfg, log_reg_dataset.k, log_reg_dataset.d)
model.train(log_reg_dataset.inputs_train,
            log_reg_dataset.targets_train,
            inputs_valid=log_reg_dataset.inputs_valid,
            targets_valid=log_reg_dataset.targets_valid
            )

confidence = model.get_model_confidence(log_reg_dataset.inputs_test)
preds = np.argmax(confidence, axis=0)

test_accuracy = accuracy(preds, log_reg_dataset.targets_test)
test_matrix = confusion_matrix(preds, log_reg_dataset.targets_test)

print(f"Test accuracy : ", test_accuracy)
print(f"Test confusion matrix : ")
print(test_matrix)

model.save('saved_models/logistic-regression.pkl')

