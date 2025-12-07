import numpy as np

class BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        # y_pred: Saída da rede (após Sigmoid), valores entre 0 e 1
        # y_true: Rótulos reais (0 ou 1)
        
        # Clip para evitar log(0) que daria -infinito e quebraria a rede
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Fórmula da Entropia Cruzada Binária
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Retorna a média do erro (para o batch ou escalar)
        return np.mean(loss)

    def backward(self):
        # Derivada da Loss em relação à previsão (y_pred)
        # dL/dy_pred = -(y/y_pred) + (1-y)/(1-y_pred)
        # Simplificando matematicamente:
        
        epsilon = 1e-15
        y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)
        
        grad_input = - (self.y_true / y_pred) + ((1 - self.y_true) / (1 - y_pred))
        
        # Se vocês estiverem calculando a média no forward, dividam por N aqui também
        return grad_input / y_pred.size