import numpy as np
#TODO camada covolucional completa
#TODO canais
#TODO funcao de ativação
#TODO backpropagation
#TODO convolução 1x1
#TODO Pooling
#TODO Fully Connected no final


class ConvLayer:
    '''Implementação sem usar matriz de pesos'''
    def __init__(self, kernel_size:int, weights:np.ndarray,out_channels:int = 1, bias:float=0, stride:int = 1, dilation:int = 1, zero_padding:int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.zero_padding = zero_padding
        self.weights = weights #por enquanto deixar salvo NOTE
        self.bias = bias
        self.out_channels = out_channels
        
        # print("peso antes")
        # print(self.weights)
                               #linha                     coluna
        res = np.zeros((out_channels, dilation * kernel_size - dilation + 1), dtype=weights.dtype) #cria uma matriz de zeros para ajustar o formato dos pesos
        # print(np.zeros)
        res[:,::dilation] = weights #preenche o vetor de zeros com os pesos pulando o valor da dilatação
        self.kernel_weights = res

        # print("peso depois")
        # print(self.kernel_weights)
        # input()
        


    
    def forward(self, input_data:np.ndarray) -> np.ndarray: #TODO depois alterar o nome
        input_used = np.pad(input_data, self.zero_padding)

        #Calculo do tamanho do resultado
        result_size = int(np.ceil((input_used.size - self.kernel_weights[0].size + 1) / self.stride))

        result = np.zeros((self.out_channels, result_size))
        
        for i in range(self.out_channels):
            for j in range(result_size):
                pos_acesso = j * self.stride
                h = np.dot(input_used[pos_acesso:pos_acesso+self.kernel_weights[i].size], self.kernel_weights[i]) + self.bias
                
                # print(f'Colocando {h} na posição {i//self.stride} do vetor {result}')
                result[i, j] = h
        
        # print('Fim:')
        # print(result)
        return result


peso = [[1,2,3],
        [4,5,6],
        [7,8,9],
        [10,11,12]]

chanels = 4


kernel = ConvLayer(kernel_size=3, weights=np.array(peso), out_channels=chanels, dilation=1, stride=1, zero_padding=1)

input_data = np.array([0,1,2,3,4,5,6,7])

print(kernel.forward(input_data))

# print(np.correlate([0,1,2,3,4,5,6], [1,2,3], 'valid'))