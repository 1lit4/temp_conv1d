from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


FREQ = 20
WINDOW_SECONDS = 10
WINDOW_SIZE = FREQ * WINDOW_SECONDS

ATIVIDADES_ESTATICAS = ['D', 'E', 'F', 'Q', 'G', 'H', 'I', 'J', 'K', 'L', 'S', 'M']


def criar_janela(pontos:np.ndarray, atividades:np.ndarray, tamanho_janela:int):
    janelas = []
    atividades_janelas = []

    for i in range(0, len(pontos) - tamanho_janela + 1, tamanho_janela):
        janela = pontos[i:i + tamanho_janela]
        janelas.append(janela)
        atividades_janelas.append(atividades[i])

    return np.array(janelas), np.array(atividades_janelas)


def ler_arquivos(lista_ids:List[int], caminho_base:str = 'dataset', atividades_estaticas:list = None, limite:int = 60, window_size:int = WINDOW_SIZE):
    if atividades_estaticas is None:
        atividades_estaticas = ATIVIDADES_ESTATICAS

    caminho_base = Path(caminho_base)
    lista_janelas = []
    lista_atividades = []

    for id_pessoa in lista_ids:
        caminho = caminho_base / 'accel' / f'data_{id_pessoa}_accel_watch.txt'
        if not caminho.exists():
            raise FileNotFoundError(f'Arquivo não encontrado: {caminho}')

        colunas = ['id_pessoa', 'atividade', 'timestamp', 'accel_x', 'accel_y', 'accel_z']
        dados = pd.read_csv(caminho, header=None, names=colunas)
        dados['accel_z'] = dados['accel_z'].astype(str).str.replace(';', '', regex=False).astype(float)

        pontos = dados[['accel_x', 'accel_y', 'accel_z']].to_numpy()
        atividades = dados['atividade'].to_numpy()
        rotulos = np.array([0 if atividade in atividades_estaticas else 1 for atividade in atividades])

        janelas, rotulos_janelas = criar_janela(pontos, rotulos, window_size)
        if len(janelas) > limite:
            indices = np.random.choice(len(janelas), limite, replace=False)
            janelas = janelas[indices]
            rotulos_janelas = rotulos_janelas[indices]

        lista_janelas.extend(janelas)
        lista_atividades.extend(rotulos_janelas)

    return np.array(lista_janelas), np.array(lista_atividades)
