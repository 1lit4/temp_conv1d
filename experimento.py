from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from Conv_1D import ConvLayer
from FullyConnected import FullyConnected
from activ_func import ReLU, Sigmoid
from flatten import Flatten
from loss_func import BinaryCrossEntropy
from max_pooling import MaxPooling
from pre_processing import ler_arquivos


TRAIN_IDS = list(range(1600, 1608))
TEST_IDS = [1608, 1609]
EPOCHS = 15
LEARNING_RATE = 0.1


class TorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 4, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(396, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)


class NumpyCNN:
    def __init__(self, torch_model):
        pesos_conv = torch_model.conv.weight.detach().numpy().transpose(0, 2, 1).copy()
        bias_conv = torch_model.conv.bias.detach().numpy().copy()
        pesos_fc = torch_model.fc.weight.detach().numpy().copy()
        bias_fc = torch_model.fc.bias.detach().numpy().copy()

        self.conv = ConvLayer(3, weights=pesos_conv, bias=bias_conv, in_channels=3, out_channels=4)
        self.relu = ReLU()
        self.pool = MaxPooling(2)
        self.flatten = Flatten()
        self.fc = FullyConnected(396, 1, weights=pesos_fc, bias=bias_fc)
        self.sigmoid = Sigmoid()
        self.loss = BinaryCrossEntropy()

    def forward(self, entrada):
        saida = self.conv.forward(entrada)
        saida = self.relu.forward(saida)
        saida = self.pool.forward(saida)
        saida = self.flatten.forward(saida)
        saida = self.fc.forward(saida)
        return self.sigmoid.forward(saida)

    def backward(self, learning_rate):
        grad = self.loss.backward()
        grad = self.sigmoid.backward(grad)
        grad = self.fc.backward(grad, learning_rate)
        grad = self.flatten.backward(grad)
        grad = self.pool.backward(grad)
        grad = self.relu.backward(grad)
        return self.conv.backward(grad, learning_rate)


def avaliar_numpy(modelo, entradas, rotulos):
    perda_total = 0
    acertos = 0

    for entrada, rotulo in zip(entradas, rotulos):
        predicao = modelo.forward(entrada)
        perda_total += float(modelo.loss.forward(predicao, rotulo))
        acertos += int((predicao.item() > 0.5) == rotulo)

    return perda_total / len(entradas), acertos / len(entradas)


def plotar_resultados(historico, caminho='resultados.png'):
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(14, 5))

    loss_ax.plot(historico['torch_train_loss'], 'r--', label='Torch Train')
    loss_ax.plot(historico['torch_test_loss'], 'r-', label='Torch Test')
    loss_ax.plot(historico['numpy_train_loss'], 'b--', label='NumPy Train')
    loss_ax.plot(historico['numpy_test_loss'], 'b-', label='NumPy Test')
    loss_ax.set_title('Loss: Treino vs Teste')
    loss_ax.set_xlabel('Épocas')
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    acc_ax.plot(historico['torch_train_acc'], 'r--', label='Torch Train')
    acc_ax.plot(historico['torch_test_acc'], 'r-', label='Torch Test')
    acc_ax.plot(historico['numpy_train_acc'], 'b--', label='NumPy Train')
    acc_ax.plot(historico['numpy_test_acc'], 'b-', label='NumPy Test')
    acc_ax.set_title('Acurácia: Treino vs Teste')
    acc_ax.set_xlabel('Épocas')
    acc_ax.legend()
    acc_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(caminho, dpi=150)
    plt.close(fig)


def executar_experimento(caminho_dataset):
    np.random.seed(42)
    torch.manual_seed(42)

    X_train, y_train = ler_arquivos(TRAIN_IDS, caminho_dataset)
    X_test, y_test = ler_arquivos(TEST_IDS, caminho_dataset)

    scaler = StandardScaler()
    quantidade, comprimento, canais = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, canais)).reshape(quantidade, comprimento, canais)
    X_test = scaler.transform(X_test.reshape(-1, canais)).reshape(len(X_test), comprimento, canais)

    torch_model = TorchCNN()
    numpy_model = NumpyCNN(torch_model)
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    X_test_torch = torch.tensor(X_test.transpose(0, 2, 1)).float()
    y_test_torch = torch.tensor(y_test).float().unsqueeze(1)

    historico = {
        'torch_train_loss': [], 'torch_test_loss': [],
        'torch_train_acc': [], 'torch_test_acc': [],
        'numpy_train_loss': [], 'numpy_test_loss': [],
        'numpy_train_acc': [], 'numpy_test_acc': []
    }

    for epoch in range(EPOCHS):
        indices = np.random.permutation(len(X_train))
        torch_model.train()
        torch_loss = 0
        torch_acertos = 0
        numpy_loss = 0
        numpy_acertos = 0

        for indice in indices:
            entrada = X_train[indice]
            rotulo = y_train[indice]

            entrada_torch = torch.tensor(entrada.T).unsqueeze(0).float()
            rotulo_torch = torch.tensor([[rotulo]]).float()
            optimizer.zero_grad()
            predicao_torch = torch_model(entrada_torch)
            perda_torch = criterion(predicao_torch, rotulo_torch)
            perda_torch.backward()
            optimizer.step()

            torch_loss += perda_torch.item()
            torch_acertos += int((predicao_torch.item() > 0.5) == rotulo)

            predicao_numpy = numpy_model.forward(entrada)
            numpy_loss += float(numpy_model.loss.forward(predicao_numpy, rotulo))
            numpy_acertos += int((predicao_numpy.item() > 0.5) == rotulo)
            numpy_model.backward(LEARNING_RATE)

        torch_model.eval()
        with torch.no_grad():
            predicao_teste_torch = torch_model(X_test_torch)
            perda_teste_torch = criterion(predicao_teste_torch, y_test_torch).item()
            acc_teste_torch = ((predicao_teste_torch > 0.5) == y_test_torch).float().mean().item()

        perda_teste_numpy, acc_teste_numpy = avaliar_numpy(numpy_model, X_test, y_test)

        historico['torch_train_loss'].append(torch_loss / len(X_train))
        historico['torch_test_loss'].append(perda_teste_torch)
        historico['torch_train_acc'].append(torch_acertos / len(X_train))
        historico['torch_test_acc'].append(acc_teste_torch)
        historico['numpy_train_loss'].append(numpy_loss / len(X_train))
        historico['numpy_test_loss'].append(perda_teste_numpy)
        historico['numpy_train_acc'].append(numpy_acertos / len(X_train))
        historico['numpy_test_acc'].append(acc_teste_numpy)

        print(
            f'Época {epoch + 1:02d} | '
            f'Torch Loss: {torch_loss / len(X_train):.3f} | '
            f'NumPy Loss: {numpy_loss / len(X_train):.3f}'
        )

    plotar_resultados(historico)
    return historico


def main():
    caminho_dataset = Path('dataset')
    if not (caminho_dataset / 'accel').is_dir():
        print('Dataset WISDM não encontrado em dataset/accel.')
        return

    executar_experimento(caminho_dataset)


if __name__ == '__main__':
    main()
