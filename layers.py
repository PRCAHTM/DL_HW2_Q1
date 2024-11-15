import torch
import torch.nn.functional as F

class FCLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCLayer, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output):
        self.grad_weights = self.input.t() @ grad_output
        self.grad_bias = grad_output.sum(0)
        grad_input = grad_output @ self.weights.t()
        return grad_input

class SigmoidLayer(torch.nn.Module):
    def forward(self, x):
        self.output = torch.sigmoid(x)
        return self.output

    def backward(self, grad_output):
        grad_input = grad_output * self.output * (1 - self.output)
        return grad_input

class ReLULayer(torch.nn.Module):
    def forward(self, x):
        self.input = x
        return F.relu(x)

    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input[self.input <= 0] = 0
        return grad_input

class DropoutLayer(torch.nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x):
        if self.training:
            self.mask = (torch.rand_like(x) > self.dropout_rate).float()
            return x * self.mask / (1.0 - self.dropout_rate)
        else:
            return x

    def backward(self, grad_output):
        return grad_output * self.mask / (1.0 - self.dropout_rate) if self.training else grad_output
