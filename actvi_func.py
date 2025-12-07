import numpy as np


class ReLU:
    def __init__(self):
        self.saved_input = None
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.saved_input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, erro: np.ndarray) -> np.ndarray:
        # O gradiente será a derivada da ReLU (1 para x > 0, 0 para x <= 0) * erro
        dX = erro.copy()
        dX[self.saved_input <= 0] = 0
        return dX
    
teste = np.array(
    [[1., 2., -1.],
     [-3., 4., 5.],
    [6., -7., 8.]]
)

relu = ReLU()
print("Input:\n", teste)
output = relu.forward(teste)
print("After ReLU Forward:\n", output)

erro = np.array(
    [[0.1, 0.2, 0.3],
     [0.4, 0.5, 0.6],
     [0.7, 0.8, 0.9]]
)
print("\nBackward:")
print(relu.backward(erro))