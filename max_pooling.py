import numpy as np

class MaxPooling:
    def __init__(self, size:int): #TODO talvez um stride no futuro
        self.size = size
    def foward(self, input_data):
        
        result_size = int(np.floor(input_data.shape[0]/self.size))
        result = np.zeros((result_size, input_data.shape[1]))
        
        for i in range(result_size):
            pos = i*self.size
            janela = input_data[pos:pos+self.size]
            result[i] = np.max(janela, axis=0)
        return result


m = MaxPooling(3)

input_data = np.array([[0, 6, 1],
                       [1, 5, 1],
                       [2, 4, 1],
                       [3, 3, 1],
                       [4, 2, 1],
                       [5, 1, 1],
                       [6, 0, 1],
                       [7, 1, 1]])

print(m.foward(input_data=input_data))