import numpy as np
#TODO conversão

class FullyConnected:
    def __init__(self,  weights:np.ndarray, input_size:np.int64, bias:np.ndarray, output_size:np.int64):
        assert weights.shape == (output_size, input_size), "Pesos não compativeis com entrada e saida"
        assert bias.shape == (output_size,), "Bias não compativel com a saida"

        self.weights = weights
        self.input_size = input_size
        self.output_size = output_size
        self.saved_input = None
        self.bias = bias 

    def forward(self, input_data:np.ndarray) -> np.ndarray:
        self.saved_input = input_data

        return self.weights @ input_data + self.bias

    def backward(self, grad_output:np.ndarray, learning_rate:float = 0.001) ->np.ndarray:
        dX = self.weights.T @ grad_output
        dW = grad_output @ self.saved_input.T
        dB = np.sum(grad_output, axis=1)

        #otimizador padrão: gradient descent
        self.weights -= learning_rate * dW
        self.bias    -= learning_rate * dB
        return  dX
    