import numpy as np
class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input_data):
        self.input_shape = input_data.shape #salvar o formato original
        # Formato de acesso: (L*C, 1) 
        return input_data.T.flatten().reshape(-1, 1) #(formato vetor coluna)

    def backward(self, grad_output):
        #Desfaz  achatamento
        grad_transposed = grad_output.reshape(self.input_shape[1], self.input_shape[0])
        return grad_transposed.T
    

if __name__ == "__main__":
    test = Flatten()
    input_data = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
    flattened = test.forward(input_data)
    print("Flattened output:")
    print(flattened)
    restored = test.backward(flattened)
    print("Restored input:")
    print(restored)
