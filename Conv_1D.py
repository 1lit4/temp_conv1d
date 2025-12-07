import numpy as np
#TODO funcao de ativação []
#TODO convolução 1x1 []
#TODO Fully Connected no final []
#TODO e quando sai do range do input? Convolucao valida e etc [?]
#TODO adicionar o bias nas coisas []
#TODO gradiente estocastico


class ConvLayer:
    '''Implementação sem usar matriz de pesos'''
    def __init__(self, kernel_size:int, weights:np.ndarray, bias:np.ndarray, in_channels:int = 1, out_channels:int = 1, stride:int = 1, dilation:int = 1, zero_padding:int = 0):
        
        #verificações temporararias
        assert weights.shape == (in_channels, kernel_size, out_channels), "Os pesos devem estar no formato ( in_chanel, kernel_size, out_chanel)"
        assert bias.shape == (out_channels,), "O bias deve estar no formato ( out_chanel, )"

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.zero_padding = zero_padding

        self.saved_input = None

        self.bias = bias # (um bias para cada filtro)
        self.in_channels = in_channels
        self.out_channels = out_channels #será a quantidade de filtros

        tamanho_linhas_dilatacao = dilation * kernel_size - dilation + 1

                 #qtd matrizes            qtd_linhas_filtro   qtdcolunas
        temp = np.zeros((out_channels, tamanho_linhas_dilatacao, in_channels), dtype=weights.dtype) #cria uma matriz de zeros para ajustar o formato dos pesos
        weights = weights.transpose(2, 1, 0)
        temp[:,::dilation,:] = weights #preenche o vetor de zeros com os pesos pulando o valor da dilatação
        self.dilated_weights = temp #pesos com a dilatação aplicada
        
        self.weights = self.dilated_weights[:, ::self.dilation, :]#pesos originais
        pass

    def forward(self, input_data:np.ndarray) -> np.ndarray: 
        assert input_data.shape[1] == self.in_channels
        
        #aplicar padding em cada canal
        input_used = np.pad(input_data, ((self.zero_padding, self.zero_padding),(0,0)))
        self.saved_input = input_used #salva o input para o backpropagation
        

        #Calculo do tamanho do resultado
        result_size = int(np.ceil((input_used.shape[0] - self.dilated_weights.shape[1] + 1) / self.stride)) #NOTE: talvez quebre
        self.result_size = result_size
        result = np.zeros((result_size, self.out_channels))

        #para cada i (linha do resultado)
        for i in range(result_size):
            pos_acesso = i * self.stride #ajusta o acesso
            for j in range(self.out_channels):
                filtro_atual = self.dilated_weights[j]
                                #recorte do input (janela)                          #kernel
                janela = input_used[pos_acesso:pos_acesso+filtro_atual.shape[0]]
                
                element_wise = janela * filtro_atual
                result[i, j] = np.sum(element_wise) 

        result += self.bias  # Adiciona o vetor bias 
        return result


    def calcular_dW(self, erros):
        #nosso resultado será:
        # dL/dW_{1,1} dL/dW'_{1,1} dL/dW''_{1,1} ...
        # dL/dW_{1,2} dL/dW'_{1,2} dL/dW''_{1,1} ...
        #    ...       ...           ...

        dW = np.zeros((self.kernel_size, self.in_channels, self.out_channels))
        n_outputs = erros.shape[0]

        for linha_wi in range(self.kernel_size):

            #onde comeca o w_i
            start_index = linha_wi * self.dilation
            
            # pega todos os inputs que w_i multiplicou
            input_slice = self.saved_input[start_index : : self.stride]
            
            # Cortamos para garantir o tamanho exato (caso sobre padding no final)
            input_slice = input_slice[:n_outputs]
            
            # Verificação de segurança
            if input_slice.shape[0] != n_outputs:
                # Isso pode acontecer se o padding/stride no forward gerou janelas que o backward não alcança
                # Geralmente resolve-se cortando o erro ou o input para o menor denominador comum
                min_len = min(input_slice.shape[0], n_outputs)
                input_slice = input_slice[:min_len]
                erros_ajustado = erros[:min_len]
                dW[linha_wi] = np.dot(input_slice.T, erros_ajustado)
            else:
                dW[linha_wi] = np.dot(input_slice.T, erros)
            
        return dW.transpose(2, 0, 1)
        
    def calcular_dX(self, erros:np.ndarray):

        dL_dX = np.zeros_like(self.saved_input) #erro em relação ao input
        qtd_linhas_errros = erros.shape[0]
        
        for i in range(qtd_linhas_errros):
            pos = i * self.stride

            #multiplicar cada elemento da linha da matriz de erros por cada matriz de pesos que os gerou e somar tudo
            resultado = np.tensordot(erros[i], self.dilated_weights, axes=([0], [0]))

            #somamos o resultado na região de onde veio o erro
            dL_dX[pos : pos + self.dilated_weights.shape[1]] += resultado
        
        
        if self.zero_padding == 0:
            return dL_dX 
        return dL_dX[self.zero_padding : -self.zero_padding] #remover o padding adicionado no forward

    def calcular_dB(self, erros):
        dB = np.sum(erros, axis=0) #soma dos erros já que dout/dB = 1
        return dB

    def backward(self, erros:np.ndarray, learning_rate:float = 0.001) -> np.ndarray:
        dW = self.calcular_dW(erros)
        dB = self.calcular_dB(erros)
        dX = self.calcular_dX(erros)

        #otimizador padrão: gradient descent
        self.weights -= learning_rate * dW
        self.bias    -= learning_rate * dB
        return dX

if __name__ == "__main__":

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


    kernel = ConvLayer(kernel_size=3, weights=np.array(peso), out_channels=chanels, in_channels=3, dilation=3, stride=1)

    input_data = np.array([[0, 6, 1],
                        [1, 5, 1],
                        [2, 4, 1],
                        [3, 3, 1],
                        [4, 2, 1],
                        [5, 1, 1],
                        [6, 0, 1],
                        [7, 1, 1]]).astype(float)

    erro = np.array([[1.,1.,1.],
                    [1.,1.,1.]])

    print(kernel.forward(input_data))


    kernel.backward(erro)

    # print(np.correlate([0,1,2,3,4,5,6], [1,2,3], 'valid'))