import Hank.nn as nn
from Hank.utils import one_hot_encoding
import numpy as np

"""
    Loss functions
    supported loss functions:
        - BCELoss
        - CrossEntropyLoss
"""
class BCELoss():
    def __init__(self):
        pass
    def __call__(self, y_pred, y_target, eps = 1e-6):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_target * np.log(y_pred) + (1 - y_target) * np.log(1 - y_pred))
        delta = -(y_target / y_pred - (1 - y_target) / (1 - y_pred))
        return loss, delta

class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, y_pred, y_target, eps = 1e-8):
        y_target = one_hot_encoding(y_target, y_pred.shape[0]).T
        y_pred = np.exp(y_pred - np.max(y_pred)) / np.sum(np.exp(y_pred - np.max(y_pred)), axis = 0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_target * np.log(y_pred))
        delta = (y_pred - y_target)
        return loss, delta