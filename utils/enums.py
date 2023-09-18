from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'valid', 'test'))
TrainType = IntEnum('TrainType', ('gradient_descent', 'normal_equation'))
