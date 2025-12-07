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