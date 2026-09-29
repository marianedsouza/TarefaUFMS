# -*- coding: utf-8 -*-
"""
Carregamento da base de 150 textos em portugues.

A base fica no arquivo `dataset_150_textos_portugues.csv` (colunas
`texto` e `categoria`), gerado de forma reprodutivel por `criar_dataset.py`.
Sao 6 categorias tematicas x 25 textos = 150.

O rotulo real (categoria) NAO e usado pelo K-Means: ele serve apenas
para, no final, avaliarmos se os grupos descobertos de forma nao
supervisionada correspondem aos temas reais.
"""

import os
import pandas as pd

CSV_PADRAO = "dataset_150_textos_portugues.csv"


def carregar_df(caminho=CSV_PADRAO):
    """Retorna o DataFrame com colunas 'texto' e 'categoria'.
    Se o CSV nao existir, gera-o automaticamente."""
    if not os.path.exists(caminho):
        import criar_dataset
        criar_dataset.main()
    return pd.read_csv(caminho, encoding="utf-8-sig")


def carregar_dados(caminho=CSV_PADRAO):
    """Retorna (lista_de_textos, lista_de_rotulos_reais)."""
    df = carregar_df(caminho)
    return df["texto"].tolist(), df["categoria"].tolist()


if __name__ == "__main__":
    from collections import Counter
    textos, rotulos = carregar_dados()
    print(f"Total de textos: {len(textos)}")
    print("Distribuicao por categoria real:", dict(Counter(rotulos)))
