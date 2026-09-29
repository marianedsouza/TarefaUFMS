"""
Atividade: Reducao de dimensionalidade (PCA) e agrupamento (K-Means)
sobre embeddings gerados por um modelo de deep learning.

Fluxo do pipeline:
    1. Gera embeddings de 80 frases com o modelo all-MiniLM-L6-v2 (384 dim).
    2. Normaliza os dados (norma L2).
    3. Aplica PCA e analisa a variancia explicada.
    4. Aplica K-Means variando k de 2 a 10 e escolhe o melhor k
       usando silhouette_score (maior valor de silhueta).
    5. Analisa e explica os grupos encontrados, comparando-os com as
       categorias reais das frases.

Saidas geradas (pasta figuras/ e resultados/):
    - figuras/pca_variancia_explicada.png
    - figuras/pca_projecao_2d_categorias.png
    - figuras/silhouette_por_k.png
    - figuras/clusters_kmeans_2d.png
    - resultados/atribuicoes.csv
    - resultados/resumo.txt
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
matplotlib.use("Agg")  # backend sem janela, apenas salva arquivos
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from dados import carregar_dados
from embeddings import gerar_embeddings

RANDOM_STATE = 42
PASTA_FIG = "figuras"
PASTA_RES = "resultados"


def preparar_pastas():
    os.makedirs(PASTA_FIG, exist_ok=True)
    os.makedirs(PASTA_RES, exist_ok=True)


# ---------------------------------------------------------------------------
# 1) PCA
# ---------------------------------------------------------------------------
def aplicar_pca(X, linhas_relatorio):
    """
    Normaliza os embeddings (norma L2, padrao recomendado para vetores de
    sentence-transformers) e aplica PCA.
    Retorna:
        X_pca_full : projecao em todas as componentes
        pca        : objeto PCA ajustado
        X_norm     : dados normalizados
    """
    print("\n=== ETAPA PCA ===")
    # Normalizacao L2: coloca todos os vetores na mesma escala (norma 1),
    # que e a forma correta de comparar embeddings de texto por distancia.
    X_norm = normalize(X)

    pca = PCA(random_state=RANDOM_STATE)
    X_pca_full = pca.fit_transform(X_norm)

    var = pca.explained_variance_ratio_
    var_acum = np.cumsum(var)

    # Quantas componentes para atingir 80% e 90% da variancia
    n_80 = int(np.argmax(var_acum >= 0.80) + 1)
    n_90 = int(np.argmax(var_acum >= 0.90) + 1)

    msg = [
        f"Dimensao original dos embeddings: {X.shape[1]}",
        f"Variancia explicada pelas 2 primeiras componentes: {var[:2].sum():.2%}",
        f"Variancia explicada pelas 3 primeiras componentes: {var[:3].sum():.2%}",
        f"Componentes necessarias para >= 80% da variancia: {n_80}",
        f"Componentes necessarias para >= 90% da variancia: {n_90}",
    ]
    for m in msg:
        print("  " + m)
    linhas_relatorio.append("== PCA ==")
    linhas_relatorio.extend(msg)

    # Grafico: variancia explicada acumulada
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(var_acum) + 1), var_acum, marker="o", markersize=3)
    plt.axhline(0.80, color="orange", linestyle="--", label="80%")
    plt.axhline(0.90, color="red", linestyle="--", label="90%")
    plt.xlabel("Numero de componentes principais")
    plt.ylabel("Variancia explicada acumulada")
    plt.title("PCA - Variancia explicada acumulada")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    caminho = os.path.join(PASTA_FIG, "pca_variancia_explicada.png")
    plt.savefig(caminho, dpi=120)
    plt.close()
    print(f"  [figura] {caminho}")

    return X_pca_full, pca, X_norm


def plot_projecao_categorias(X_pca_full, rotulos):
    """Projeta em 2D (PC1 x PC2) colorindo pelas categorias reais."""
    df = pd.DataFrame({
        "PC1": X_pca_full[:, 0],
        "PC2": X_pca_full[:, 1],
        "categoria": rotulos,
    })
    plt.figure(figsize=(8, 6))
    for cat in sorted(df["categoria"].unique()):
        sub = df[df["categoria"] == cat]
        plt.scatter(sub["PC1"], sub["PC2"], label=cat, s=40, alpha=0.8)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Projecao PCA 2D - cor = categoria real")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    caminho = os.path.join(PASTA_FIG, "pca_projecao_2d_categorias.png")
    plt.savefig(caminho, dpi=120)
    plt.close()
    print(f"  [figura] {caminho}")


# ---------------------------------------------------------------------------
# 2) K-Means + silhouette_score
# ---------------------------------------------------------------------------
def escolher_k(X, linhas_relatorio, k_min=2, k_max=10):
    """
    Aplica K-Means para cada k em [k_min, k_max] e calcula o
    silhouette_score. Retorna o k com maior valor de silhueta.
    """
    print("\n=== ESCOLHA DO NUMERO DE GRUPOS (silhouette_score) ===")
    ks = list(range(k_min, k_max + 1))
    scores = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        scores.append(s)
        print(f"  k={k:2d}  silhouette_score={s:.4f}")
        linhas_relatorio.append(f"k={k}: silhouette={s:.4f}")

    melhor_idx = int(np.argmax(scores))
    melhor_k = ks[melhor_idx]
    melhor_score = scores[melhor_idx]

    print(f"\n  >> Melhor k = {melhor_k} (silhouette = {melhor_score:.4f})")
    linhas_relatorio.append(f"MELHOR k = {melhor_k} (silhouette = {melhor_score:.4f})")

    # Grafico silhouette x k
    plt.figure(figsize=(8, 5))
    plt.plot(ks, scores, marker="o")
    plt.scatter([melhor_k], [melhor_score], color="red", zorder=5,
                label=f"melhor k = {melhor_k}")
    plt.xlabel("Numero de grupos (k)")
    plt.ylabel("silhouette_score")
    plt.title("Escolha de k pelo coeficiente de silhueta")
    plt.xticks(ks)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    caminho = os.path.join(PASTA_FIG, "silhouette_por_k.png")
    plt.savefig(caminho, dpi=120)
    plt.close()
    print(f"  [figura] {caminho}")

    return melhor_k, dict(zip(ks, scores))


def aplicar_kmeans_final(X, k):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


def plot_clusters_2d(X_pca_full, labels, k, centroides=None):
    """Projeta em 2D colorindo pelos clusters do K-Means."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca_full[:, 0], X_pca_full[:, 1],
                          c=labels, cmap="tab10", s=45, alpha=0.85)
    if centroides is not None and centroides.shape[1] >= 2:
        plt.scatter(centroides[:, 0], centroides[:, 1],
                    c="black", marker="X", s=180, edgecolors="white",
                    label="centroides")
        plt.legend()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Grupos do K-Means (k={k}) na projecao PCA 2D")
    plt.colorbar(scatter, label="cluster")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    caminho = os.path.join(PASTA_FIG, "clusters_kmeans_2d.png")
    plt.savefig(caminho, dpi=120)
    plt.close()
    print(f"  [figura] {caminho}")


# ---------------------------------------------------------------------------
# 3) Analise dos grupos
# ---------------------------------------------------------------------------
def analisar_grupos(textos, rotulos, labels, linhas_relatorio):
    print("\n=== ANALISE DOS GRUPOS ENCONTRADOS ===")
    df = pd.DataFrame({"texto": textos, "categoria_real": rotulos, "cluster": labels})

    linhas_relatorio.append("")
    linhas_relatorio.append("== ANALISE DOS GRUPOS ==")

    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        contagem = Counter(sub["categoria_real"])
        categoria_dominante, qtd = contagem.most_common(1)[0]
        pureza = qtd / len(sub)

        cab = (f"\nCluster {c}: {len(sub)} frases | "
               f"categoria dominante = '{categoria_dominante}' "
               f"({qtd}/{len(sub)} = {pureza:.0%} de pureza)")
        print(cab)
        print("  composicao:", dict(contagem))
        print("  exemplos:")
        for txt in sub["texto"].head(3):
            print(f"    - {txt}")

        linhas_relatorio.append(cab.strip())
        linhas_relatorio.append(f"  composicao: {dict(contagem)}")
        for txt in sub["texto"].head(3):
            linhas_relatorio.append(f"  exemplo: {txt}")

    # Concordancia global entre clusters e categorias reais
    ari = adjusted_rand_score(rotulos, labels)
    print(f"\n  Adjusted Rand Index (clusters vs categorias reais): {ari:.4f}")
    linhas_relatorio.append(f"\nAdjusted Rand Index (clusters vs categorias reais): {ari:.4f}")

    # Salva CSV com todas as atribuicoes
    caminho_csv = os.path.join(PASTA_RES, "atribuicoes.csv")
    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
    print(f"  [csv] {caminho_csv}")

    return df, ari


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main():
    preparar_pastas()
    linhas_relatorio = []

    # 0) Dados + embeddings de deep learning
    textos, rotulos = carregar_dados()
    print(f"Total de frases: {len(textos)}")
    print("Categorias reais:", dict(Counter(rotulos)))
    X, info = gerar_embeddings(textos)
    linhas_relatorio.append(f"Modelo de deep learning: {info['modelo']}")
    linhas_relatorio.append(f"Dimensao dos embeddings: {info['dimensao']}")
    linhas_relatorio.append(f"Numero de frases: {len(textos)}")

    # 1) PCA
    X_pca_full, pca, X_norm = aplicar_pca(X, linhas_relatorio)
    plot_projecao_categorias(X_pca_full, rotulos)

    # Reducao para o clustering.
    # Observacao importante: rodar K-Means/silhueta diretamente nas 384
    # dimensoes (ou em dezenas de componentes) sofre com a "maldicao da
    # dimensionalidade" -- as distancias euclidianas ficam parecidas entre
    # si e o silhouette_score cai para perto de zero. Por isso reduzimos a
    # poucas componentes principais (as que concentram a maior parte da
    # estrutura dos dados) ANTES de agrupar. Aqui usamos 2 componentes.
    N_COMP_CLUSTER = 2
    n_comp = min(N_COMP_CLUSTER, X_pca_full.shape[1])
    X_reduzido = X_pca_full[:, :n_comp]
    var_usada = pca.explained_variance_ratio_[:n_comp].sum()
    print(f"\nUsando {n_comp} componentes principais para o K-Means "
          f"({var_usada:.2%} da variancia).")
    linhas_relatorio.append(
        f"Componentes usadas no K-Means: {n_comp} "
        f"({var_usada:.2%} da variancia)")

    # 2) K-Means + silhouette
    melhor_k, scores = escolher_k(X_reduzido, linhas_relatorio)
    labels, km = aplicar_kmeans_final(X_reduzido, melhor_k)
    plot_clusters_2d(X_pca_full, labels, melhor_k, km.cluster_centers_)

    # 3) Analise dos grupos
    analisar_grupos(textos, rotulos, labels, linhas_relatorio)

    # Salva resumo textual
    caminho_resumo = os.path.join(PASTA_RES, "resumo.txt")
    with open(caminho_resumo, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas_relatorio))
    print(f"\n[resumo] {caminho_resumo}")
    print("\nConcluido.")


if __name__ == "__main__":
    main()
