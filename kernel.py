import numpy as np
#TODO camada covolucional completa
#TODO canais
#TODO funcao de ativação
#TODO backpropagation
#TODO convolução 1x1
#TODO Pooling
#TODO Fully Connected no final


class Kernel:
    '''Implementação sem usar matriz de pesos'''
    def __init__(self, kernel_size:int, weights:np.ndarray, bias:float=0, stride:int = 1, dilation:int = 1, zero_padding:int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.zero_padding = zero_padding
        self.weights = weights #por enquanto deixar salvo NOTE
        self.bias = bias

        res = np.zeros(dilation * kernel_size - dilation + 1, dtype=weights.dtype) #cria um vetor de zeros
        res[::dilation] = weights #preenche o vetor de zeros com os pesos pulando o valor da dilatação
        self.kernel_weights = res


    
    def forward(self, input_data:np.ndarray) -> np.ndarray: #TODO depois alterar o nome
        input_used = np.pad(input_data, self.zero_padding)

        #Calculo do tamanho do resultado
        result_size = int(np.ceil((input_used.size - self.kernel_weights.size + 1) / self.stride))

        result = np.zeros(result_size)
        
        for i in range(0, result_size):
            pos_acesso = i * self.stride
            h = np.dot(input_used[pos_acesso:pos_acesso+self.kernel_weights.size],
                        self.kernel_weights
                        ) + self.bias
            
            # print(f'Colocando {h} na posição {i//self.stride} do vetor {result}')
            result[i] = h
        
        # print('Fim:')
        # print(result)
        return result



kernel = Kernel(kernel_size=3, weights=np.array([1,2,3]), stride=3, zero_padding=2, dilation=2)

input_data = np.array([0,1,2,3,4,5,6,7])

print(kernel.forward(input_data))

# print(np.correlate([0,1,2,3,4,5,6], [1,2,3], 'valid'))