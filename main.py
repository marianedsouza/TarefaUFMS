# -*- coding: utf-8 -*-
"""
Analise exploratoria e agrupamento de embeddings de texto em portugues.

Segue a pratica da aula:
    texto -> embedding (deep learning) -> normalizacao L2 -> PCA (2D e 3D)
          -> K-Means -> escolha de k (cotovelo + silhouette_score)
          -> comparacao com categorias reais -> caracterizacao dos grupos

Pergunta de investigacao:
    "Os embeddings semanticos organizam os textos de acordo com seus temas,
     de modo que o K-Means recupere grupos parecidos com as categorias reais?"

Observacao metodologica:
    Assim como no notebook da aula, o K-Means e treinado no ESPACO COMPLETO
    dos embeddings normalizados (384 dimensoes). O PCA e usado para
    visualizacao (2D e 3D) e a silhueta e calculada no mesmo espaco do
    agrupamento. As categorias reais NAO sao usadas pelo K-Means.

Saidas (figuras/ e resultados/):
    figuras/pca_2d_sem_rotulo.png
    figuras/pca_2d_categorias.png
    figuras/pca_2d_clusters.png
    figuras/pca_3d_clusters.png
    figuras/metodo_cotovelo.png
    figuras/silhouette_por_k.png
    resultados/atribuicoes.csv
    resultados/crosstab_categoria_cluster.csv
    resultados/resumo.txt
"""

import os
import warnings
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")

from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registra projecao 3d)

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from dados import carregar_df
from embeddings import gerar_embeddings

RANDOM_STATE = 42
PASTA_FIG = "figuras"
PASTA_RES = "resultados"
CORES = ["#2563eb", "#16a34a", "#ea580c", "#9333ea", "#dc2626",
         "#0891b2", "#ca8a04", "#db2777", "#4b5563", "#65a30d"]


def preparar_pastas():
    os.makedirs(PASTA_FIG, exist_ok=True)
    os.makedirs(PASTA_RES, exist_ok=True)


# ---------------------------------------------------------------------------
# PCA (variancia + projecoes 2D/3D)
# ---------------------------------------------------------------------------
def aplicar_pca(X, rel):
    """PCA de 3 componentes para visualizacao. Retorna X_pca (n,3) e o pca."""
    print("\n=== PCA (visualizacao) ===")
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)

    var = pca.explained_variance_ratio_
    print(f"  PC1: {var[0]:.2%} | PC2: {var[1]:.2%} | PC3: {var[2]:.2%}")
    print(f"  Variancia acumulada (3 PCs): {var.sum():.2%}")
    rel.append("== PCA (visualizacao) ==")
    rel.append(f"PC1={var[0]:.2%}, PC2={var[1]:.2%}, PC3={var[2]:.2%}")
    rel.append(f"Variancia acumulada 2 PCs: {var[:2].sum():.2%}")
    rel.append(f"Variancia acumulada 3 PCs: {var.sum():.2%}")
    return X_pca, pca


def plot_pca_2d_sem_rotulo(X_pca):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=40, alpha=0.75, color="#334155")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Embeddings de texto projetados com PCA (2D) - sem rotulos")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _salvar("pca_2d_sem_rotulo.png")


def plot_pca_2d_categorias(X_pca, categorias):
    categorias = np.array(categorias)
    plt.figure(figsize=(8, 6))
    for i, cat in enumerate(sorted(set(categorias))):
        m = categorias == cat
        plt.scatter(X_pca[m, 0], X_pca[m, 1], s=40, alpha=0.8,
                    color=CORES[i % len(CORES)], label=cat)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA 2D - cor = categoria real")
    plt.legend(title="categoria", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _salvar("pca_2d_categorias.png")


def plot_pca_2d_clusters(X_pca, labels, k):
    plt.figure(figsize=(8, 6))
    for c in sorted(set(labels)):
        m = labels == c
        plt.scatter(X_pca[m, 0], X_pca[m, 1], s=40, alpha=0.8,
                    color=CORES[c % len(CORES)], label=f"cluster {c}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA 2D - cor = cluster do K-Means (k={k})")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _salvar("pca_2d_clusters.png")


def plot_pca_3d_clusters(X_pca, labels, k):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(X_pca[m, 0], X_pca[m, 1], X_pca[m, 2], s=35, alpha=0.8,
                   color=CORES[c % len(CORES)], label=f"cluster {c}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"PCA 3D - clusters do K-Means (k={k})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    _salvar("pca_3d_clusters.png")


def _salvar(nome):
    caminho = os.path.join(PASTA_FIG, nome)
    plt.savefig(caminho, dpi=120)
    plt.close()
    print(f"  [figura] {caminho}")


# ---------------------------------------------------------------------------
# Escolha de k: metodo do cotovelo + silhouette
# ---------------------------------------------------------------------------
def escolher_k(X, rel, k_min=2, k_max=10):
    print("\n=== ESCOLHA DE k (cotovelo + silhouette_score) ===")
    ks = list(range(k_min, k_max + 1))
    inercias, silhuetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inercias.append(km.inertia_)
        s = silhouette_score(X, labels, metric="euclidean")
        silhuetas.append(s)
        print(f"  k={k:2d}  inercia={km.inertia_:8.2f}  silhouette={s:.4f}")
        rel.append(f"k={k}: inercia={km.inertia_:.2f}, silhouette={s:.4f}")

    melhor_idx = int(np.argmax(silhuetas))
    melhor_k = ks[melhor_idx]
    melhor_s = silhuetas[melhor_idx]
    print(f"\n  >> Melhor k pela silhueta = {melhor_k} (silhouette = {melhor_s:.4f})")
    rel.append(f"MELHOR k (silhouette) = {melhor_k} (silhouette = {melhor_s:.4f})")

    # Grafico do cotovelo
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inercias, marker="o")
    plt.xlabel("Numero de grupos (k)")
    plt.ylabel("Inercia (soma das distancias intra-cluster)")
    plt.title("Metodo do cotovelo")
    plt.xticks(ks)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _salvar("metodo_cotovelo.png")

    # Grafico da silhueta
    plt.figure(figsize=(8, 5))
    plt.plot(ks, silhuetas, marker="o")
    plt.scatter([melhor_k], [melhor_s], color="red", zorder=5,
                label=f"melhor k = {melhor_k}")
    plt.xlabel("Numero de grupos (k)")
    plt.ylabel("silhouette_score")
    plt.title("Silhouette Score por numero de grupos")
    plt.xticks(ks)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _salvar("silhouette_por_k.png")

    return melhor_k, dict(zip(ks, silhuetas)), dict(zip(ks, inercias))


# ---------------------------------------------------------------------------
# Analise / caracterizacao dos grupos
# ---------------------------------------------------------------------------
def analisar_grupos(df, rel):
    print("\n=== CARACTERIZACAO DOS GRUPOS ===")
    rel.append("")
    rel.append("== CARACTERIZACAO DOS GRUPOS ==")

    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        contagem = Counter(sub["categoria"])
        dominante, qtd = contagem.most_common(1)[0]
        pureza = qtd / len(sub)
        cab = (f"\nCluster {c}: {len(sub)} textos | dominante = '{dominante}' "
               f"({qtd}/{len(sub)} = {pureza:.0%})")
        print(cab)
        print("  composicao:", dict(contagem))
        for txt in sub["texto"].head(3):
            print(f"    - {txt}")
        rel.append(cab.strip())
        rel.append(f"  composicao: {dict(contagem)}")
        for txt in sub["texto"].head(3):
            rel.append(f"  exemplo: {txt}")

    # Crosstab categoria x cluster
    ct = pd.crosstab(df["categoria"], df["cluster"])
    print("\nTabela de contingencia (categoria x cluster):")
    print(ct)
    ct.to_csv(os.path.join(PASTA_RES, "crosstab_categoria_cluster.csv"),
              encoding="utf-8-sig")

    ari = adjusted_rand_score(df["categoria"], df["cluster"])
    print(f"\n  Adjusted Rand Index (clusters vs categorias): {ari:.4f}")
    rel.append(f"\nTabela de contingencia:\n{ct.to_string()}")
    rel.append(f"\nAdjusted Rand Index: {ari:.4f}")

    df[["texto", "categoria", "cluster"]].to_csv(
        os.path.join(PASTA_RES, "atribuicoes.csv"),
        index=False, encoding="utf-8-sig")
    return ari


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main():
    preparar_pastas()
    rel = []

    # 0) Dados + embeddings
    df = carregar_df()
    print(f"Total de textos: {len(df)}")
    print("Categorias reais:", dict(Counter(df["categoria"])))
    emb, info = gerar_embeddings(df["texto"].tolist())
    rel.append(f"Modelo de deep learning: {info['modelo']}")
    rel.append(f"Dimensao dos embeddings: {info['dimensao']}")
    rel.append(f"Numero de textos: {len(df)}")
    rel.append(f"Categorias: {sorted(df['categoria'].unique())}")

    # 1) Normalizacao L2 (padrao para embeddings semanticos)
    X = normalize(emb, norm="l2")
    print(f"Norma do primeiro vetor apos L2: {np.linalg.norm(X[0]):.4f}")

    # 2) PCA para visualizacao (2D e 3D)
    X_pca, pca = aplicar_pca(X, rel)
    plot_pca_2d_sem_rotulo(X_pca)
    plot_pca_2d_categorias(X_pca, df["categoria"].tolist())

    # 3) Escolha de k (cotovelo + silhueta) no espaco COMPLETO dos embeddings
    melhor_k, silhuetas, inercias = escolher_k(X, rel)

    # 4) K-Means final com o melhor k
    km = KMeans(n_clusters=melhor_k, random_state=RANDOM_STATE, n_init=10)
    df["cluster"] = km.fit_predict(X)

    # 5) Visualizacao dos clusters (2D e 3D)
    plot_pca_2d_clusters(X_pca, df["cluster"].values, melhor_k)
    plot_pca_3d_clusters(X_pca, df["cluster"].values, melhor_k)

    # 6) Caracterizacao dos grupos + comparacao com categorias reais
    analisar_grupos(df, rel)

    with open(os.path.join(PASTA_RES, "resumo.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rel))
    print(f"\n[resumo] {os.path.join(PASTA_RES, 'resumo.txt')}")
    print("Concluido.")


if __name__ == "__main__":
    main()
