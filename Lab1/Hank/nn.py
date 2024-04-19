import numpy as np
from abc import ABC, abstractmethod


"""
    definition of module 
"""
class module(ABC):
    def __init__(self):
        self.net = None
        self.delta = None
        pass
    def __call__(self, x):
        return self.forward(x)
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def backward(self):
        pass

"""
    definition of activation functions
    supported activation functions:
        - sigmoid
        - ReLU
        - LeakyReLU
        - Tanh
        - Identity
"""
activation_functions_list = ["sigmoid", "ReLU", "LeakyReLU", "Tanh", "Identity"]

class activation(ABC):
    def __init__(self):
        self.a = None
    @abstractmethod
    def forward(self):
        pass
    @abstractmethod
    def backward(self):
        pass
    
class sigmoid(activation):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        self.a = x
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward(self, x):
        return np.multiply(x, 1 - x)

class ReLU(activation):
    def forward(self, x):
        self.a = x
        return np.maximum(0, x)
    
    def backward(self, x):
        return (x > 0).astype(int)

class LeakyReLU(activation):
    def __init__(self, alpha = 0.02):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        self.a = x
        return np.maximum(self.alpha * x, x)
    
    def backward(self, x):
        return (x > 0).astype(int) + self.alpha * (x <= 0).astype(int)
    
class Tanh(activation):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        super().__init__()
        self.a = x
        return np.tanh(x)
    
    def backward(self, x):
        return 1 - x ** 2
    
class Identity(activation):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        self.a = x
        return x
    
    def backward(self, x):
        return np.ones_like(x)

class Softmax(activation):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        self.a = x
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis = 0)
    
    def backward(self, x):
        return x * (1 - x)

"""
    definition of layers
    supported layers:
        - Linear
"""
class layer(ABC):
    def __init__(self):
        self.w = None
        self.b = None
        self.a = None
        
    @abstractmethod
    def forward(self):
        pass
    
    @abstractmethod
    def backward(self):
        pass

class Linear(layer):
    def __init__(self, in_features, out_features):
        self.w = np.random.randn(in_features, out_features).T
        self.b = np.random.randn(1, out_features).T
        self.a = None
        
    def forward(self, x):
        self.a = x
        return self.w @ x + self.b

    def backward(self, delta):
        return self.w.T @ delta
    
class Simple_Conv1d(layer):
    def __init__(self, kernel_size = 3):
        self.w = np.ones((kernel_size, 1))
        self.b = 0

    def forward(self, x):
        h = np.zeros(x.shape[-2] - self.w.shape[0] + 1)
        for i in range(h.shape[0]):
            h[i] = np.sum(x[i:i + self.w.shape[0]] * self.w) + self.b
        return self.act.forward(h)
