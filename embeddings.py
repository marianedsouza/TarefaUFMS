"""
Geracao de embeddings (vetores de caracteristicas) com um modelo de
deep learning.

Usamos o modelo multilingue pre-treinado
'paraphrase-multilingual-MiniLM-L12-v2' da biblioteca sentence-transformers.
Ele e uma rede neural do tipo Transformer que suporta portugues e converte
cada frase em um vetor denso de 384 dimensoes, no qual frases com
significado parecido ficam proximas no espaco vetorial.

Esses vetores de alta dimensao (384) sao exatamente o tipo de dado em
que a reducao de dimensionalidade com PCA faz sentido.
"""

import os
import warnings

# Silencia avisos inofensivos de cache/symlink do Hugging Face no Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

import numpy as np

MODELO_PADRAO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def gerar_embeddings(textos, nome_modelo=MODELO_PADRAO):
    """
    Converte uma lista de textos em uma matriz de embeddings.

    Retorna:
        emb: np.ndarray de shape (n_textos, n_dimensoes)
        info: dict com metadados (modelo usado, dimensao)
    """
    from sentence_transformers import SentenceTransformer

    print(f"Carregando modelo de deep learning: {nome_modelo} ...")
    modelo = SentenceTransformer(nome_modelo)
    print("Gerando embeddings das frases (isso passa cada frase pela rede neural)...")
    emb = modelo.encode(textos, show_progress_bar=False, convert_to_numpy=True)
    emb = np.asarray(emb, dtype=np.float64)

    info = {"modelo": nome_modelo, "dimensao": emb.shape[1]}
    print(f"Embeddings gerados: {emb.shape[0]} frases x {emb.shape[1]} dimensoes.")
    return emb, info


if __name__ == "__main__":
    from dados import carregar_dados

    textos, _ = carregar_dados()
    emb, info = gerar_embeddings(textos)
    print(info)
    print("Primeiro vetor (5 primeiras dimensoes):", emb[0][:5])
