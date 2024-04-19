import Hank.nn as nn
import numpy as np
    
"""
    supported learning rate scheduler:
        - StepRL
"""
class StepRL():
    def __init__(self, optimizer, step_size, gamma = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.t = 0
    
    def step(self):
        self.t += 1
        if self.t % self.step_size == 0:
            self.optimizer.lr *= self.gamma

"""
    supported optimizers:
        - SGD
        - SGDM
        - RMSprop
        - Adam
"""

optimier_list = ["SGD", "SGDM", "RMSProp", "Adam"]

class SGD():
    def __init__(self, model, lr = 0.001):
        self.lr = lr
        self.model = model
    
    def step(self):
        for i, layer in enumerate(self.model.net):
            if isinstance(layer, nn.layer):
                layer.w -= self.lr * (self.model.delta[i + 1] @ layer.a.T)
                layer.b -= self.lr * self.model.delta[i + 1]

class SGDM():
    def __init__(self, model, lr = 0.001, momentum = 0.9):
        assert isinstance(model, nn.module), "model should be an instance of module"
        assert 0 < momentum < 1, "momentum should be in (0, 1)"
        self.lr = lr
        self.momentum = momentum
        self.model = model
        self.v_w = []
        self.v_b = []
        for layer in self.model.net:
            if isinstance(layer, nn.layer):
                self.v_w.append(np.zeros_like(layer.w))
                self.v_b.append(np.zeros_like(layer.b))
            else:
                self.v_w.append(None)
                self.v_b.append(None)
                
    def step(self):
        for i, layer in enumerate(self.model.net):
            if isinstance(layer, nn.layer):
                self.v_w[i] = self.momentum * self.v_w[i] - self.lr * self.model.delta[i + 1] @ layer.a.T
                self.v_b[i] = self.momentum * self.v_b[i] - self.lr * self.model.delta[i + 1]
                layer.w += self.v_w[i]
                layer.b += self.v_b[i]

class RMSprop():
    def __init__(self, model, lr = 0.001, alpha = 0.9, epsilon = 1e-8):
        assert isinstance(model, nn.module), "model should be an instance of module"
        assert 0 < alpha < 1, "alpha should be in (0, 1)"
        assert 0 < epsilon, "epsilon should be positive"
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.model = model
        self.v_w = []
        self.v_b = []
        for layer in self.model.net:
            if isinstance(layer, nn.layer):
                self.v_w.append(np.zeros_like(layer.w))
                self.v_b.append(np.zeros_like(layer.b))
            else:
                self.v_w.append(None)
                self.v_b.append(None)
        
    def step(self):
        for i, layer in enumerate(self.model.net):
            if isinstance(layer, nn.layer):
                self.v_w[i] = self.alpha * self.v_w[i] + (1 - self.alpha) * (self.model.delta[i + 1] @ layer.a.T) ** 2
                self.v_b[i] = self.alpha * self.v_b[i] + (1 - self.alpha) * (self.model.delta[i + 1]) ** 2
                layer.w -= self.lr * self.model.delta[i + 1] @ layer.a.T / (np.sqrt(self.v_w[i]) + self.epsilon)
                layer.b -= self.lr * self.model.delta[i + 1] / (np.sqrt(self.v_b[i]) + self.epsilon)
            
class Adam():
    def __init__(self, model, lr = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        assert isinstance(model, nn.module), "model should be an instance of module"
        assert 0 < beta1 < 1 and 0 < beta2 < 1, "beta1 and beta2 should be in (0, 1)"
        assert 0 < epsilon, "epsilon should be positive"
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.model = model
        self.m_w, self.v_w, self.m_b, self.v_b = [], [], [], []
        for layer in self.model.net:
            if isinstance(layer, nn.layer):
                self.m_w.append(np.zeros_like(layer.w))
                self.v_w.append(np.zeros_like(layer.w))
                self.m_b.append(np.zeros_like(layer.b))
                self.v_b.append(np.zeros_like(layer.b))
            else:
                self.m_w.append(None)
                self.v_w.append(None)
                self.m_b.append(None)
                self.v_b.append(None)
        self.t = 0
        
    def step(self):
        self.t += 1
        for i, layer in enumerate(self.model.net):
            if isinstance(layer, nn.layer):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * (self.model.delta[i + 1] @ layer.a.T)
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (self.model.delta[i + 1] @ layer.a.T) ** 2
                m_hat_w = self.m_w[i] / (1 - self.beta1 ** self.t)
                v_hat_w = self.v_w[i] / (1 - self.beta2 ** self.t)
                layer.w -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * self.model.delta[i + 1]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (self.model.delta[i + 1]) ** 2
                m_hat_b = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)
                layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)