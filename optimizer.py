import torch

class SGD:
    def __init__(self, params, learning_rate=0.01):
        self.params = list(params)
        self.learning_rate = learning_rate

    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.learning_rate * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
