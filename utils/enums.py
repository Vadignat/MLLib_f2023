from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'valid', 'test'))
TrainType = IntEnum('TrainType', ('gradient_descent', 'normal_equation'))
WeightsInitType = IntEnum('WeightsInitType', ('xavier_normal', 'xavier_uniform', 'he_normal', 'he_uniform'))
GDStoppingCriteria = IntEnum('GDStoppingCriteria', ('epoch', 'gradient_norm', 'difference_norm', 'metric_value'))
