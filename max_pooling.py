import numpy as np

class MaxPooling:
    def __init__(self, size:int): 
        self.size = size # tamanho da janela do max pooling
        self.saved_input = None
        self.mask = None

    def forward(self, input_data):
        self.saved_input = input_data
        result_size = int(np.floor(input_data.shape[0]/self.size))
        result = np.zeros((result_size, input_data.shape[1]))

        self.mask = np.zeros_like(input_data)
        
        for i in range(result_size):
            pos = i*self.size
            janela = input_data[pos:pos+self.size]
            result[i] = np.max(janela, axis=0)
            
            #pegar os indices dos maximos
            idx_max = janela.argmax(axis=0)
            #colocar 1 na posicao dos maximos na mask
            self.mask[pos + idx_max, np.arange(input_data.shape[1])] = 1
            
        return result
    
    def backward(self, d_output):
        d_output = np.repeat(d_output, self.size, axis = 0)

        dX = np.zeros_like(self.saved_input) #do tamanho exato da entrada origianl
        limite_preenchimento = min(d_output.shape[0], dX.shape[0])
        dX[:limite_preenchimento] = d_output[:limite_preenchimento]
        dX = dX*self.mask
        return dX
        
    
if __name__ == "__main__":

    m = MaxPooling(3)

    input_data = np.array([[0, 6, 1],
                        [1, 5, 1],
                        [2, 4, 1],
                        [3, 3, 1],
                        [4, 2, 1],
                        [5, 1, 1],
                        [6, 0, 1],
                        [7, 1, 1]])


    print(m.forward(input_data=input_data))