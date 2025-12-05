import numpy as np
#TODO camada covolucional completa [x]
#TODO canais [x]
#TODO funcao de ativação []
#TODO backpropagation []
#TODO convolução 1x1 []
#TODO Pooling []
#TODO Fully Connected no final []
#TODO e quando sai do range do input? Convolucao valida e etc [?]
#TODO adicionar o bias nas coisas []


class ConvLayer:
    '''Implementação sem usar matriz de pesos'''
    def __init__(self, kernel_size:int, weights:np.ndarray,in_channels:int = 1, out_channels:int = 1, bias:float=0, stride:int = 1, dilation:int = 1, zero_padding:int = 0):
        
        #verificações temporararias
        assert weights.shape == (in_channels, kernel_size, out_channels), "Os pesos devem estar no formato ( in_chanel, kernel_size, out_chanel)"
        
        
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.zero_padding = zero_padding
        self.weights = weights #por enquanto deixar salvo NOTE
        self.bias = bias
        self.in_channels = in_channels
        self.out_channels = out_channels

                 #qtd matrizes           tamanho coluna filtro(linhas)         colunas
        res = np.zeros((out_channels, dilation * kernel_size - dilation + 1, in_channels), dtype=weights.dtype) #cria uma matriz de zeros para ajustar o formato dos pesos
        # print(np.zeros)
        res[:,::dilation,:] = weights #preenche o vetor de zeros com os pesos pulando o valor da dilatação
        self.kernel_weights = res

    
    def forward(self, input_data:np.ndarray) -> np.ndarray: #TODO depois alterar o nome
        assert input_data.shape[1] == self.in_channels
        
        #aplicar padding em cada canal
        input_used = np.pad(input_data, ((self.zero_padding, self.zero_padding),(0,0)))
        

        #Calculo do tamanho do resultado
        result_size = int(np.ceil((input_used.shape[0] - self.kernel_weights.shape[1] + 1) / self.stride)) #NOTE: talvez quebre
        result = np.zeros((result_size, self.out_channels))

        #para cada i (linha do resultado)
        for i in range(result_size):
            pos_acesso = i * self.stride #ajusta o acesso
            for j in range(self.out_channels):
                filtro_atual = self.kernel_weights[j]
                                #recorte do input (janela)                          #kernel
                janela = input_used[pos_acesso:pos_acesso+filtro_atual.shape[0]]
                
                element_wise = janela * filtro_atual
                result[i, j] = np.sum(element_wise)
    
        return result

        #kernel size ->  #chanels
peso = [[[1,2,3],         #^
         [4,5,6],         #|
         [7,8,9]],        #
        
        [[2,3,4],         # depth 2
         [5,6,7],
         [8,9,1]],
        
        [[3,4,5],         # depth 3
         [6,7,8],
         [9,1,2]]]        #


chanels = 3
chanels_in = 3


kernel = ConvLayer(kernel_size=3, weights=np.array(peso), out_channels=chanels, in_channels=3, dilation=1, stride=1)

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