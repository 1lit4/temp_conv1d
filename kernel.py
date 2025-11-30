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
    def __init__(self, kernel_size:int, weights:np.ndarray,in_channels:int = 1, out_channels:int = 1, bias:float=0, stride:int = 1, dilation:int = 1, zero_padding:int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.zero_padding = zero_padding
        self.weights = weights #por enquanto deixar salvo NOTE
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels

        #                linha                     coluna
        res = np.zeros((dilation * kernel_size - dilation + 1, out_channels), dtype=weights.dtype) #cria uma matriz de zeros para ajustar o formato dos pesos
        # print(np.zeros)
        res[::dilation,:] = weights #preenche o vetor de zeros com os pesos pulando o valor da dilatação
        self.kernel_weights = res

    
    def forward(self, input_data:np.ndarray) -> np.ndarray: #TODO depois alterar o nome
        #aplicar padding em cada canal
        input_used = np.pad(input_data, ((self.zero_padding, self.zero_padding),(0,0)))
        

        #Calculo do tamanho do resultado
        result_size = int(np.ceil((input_used.shape[0] - self.kernel_weights.shape[0] + 1) / self.stride))
        result = np.zeros((result_size, self.out_channels))

        for i in range(result.shape[0]):
            pos_acesso = i * self.stride

                            #recorte do input                                    #kernel
            element_wise = input_used[pos_acesso:pos_acesso+self.kernel_weights.shape[0]] * self.kernel_weights
            result[i] = np.sum(element_wise, axis=1)
    
        return result

        #kernel size ->  #chanels
peso = [[1,2,3],         #^
        [4,5,6],         #|
        [7,8,9]]         #


chanels = 3
chanels_in = 3


kernel = ConvLayer(kernel_size=3, weights=np.array(peso), out_channels=chanels, dilation=1, stride=1)

input_data = np.array([[0, 6, 1],
                       [1, 5, 1],
                       [2, 4, 1],
                       [3, 3, 1],
                       [4, 2, 1],
                       [5, 1, 1],
                       [6, 0, 1],
                       [7, 1, 1]])

print(kernel.forward(input_data))

# print(np.correlate([0,1,2,3,4,5,6], [1,2,3], 'valid'))