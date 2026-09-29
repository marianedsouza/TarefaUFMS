# PCA + K-Means sobre Embeddings de Deep Learning

Documentação técnica do projeto. Ele gera representações vetoriais
(*embeddings*) de frases usando um modelo de deep learning, reduz a
dimensionalidade com **PCA** e descobre grupos com **K-Means**, escolhendo o
número de grupos pelo **coeficiente de silhueta** (`silhouette_score`).

> Para a leitura orientada à atividade (metodologia, resultados e discussão
> acadêmica), veja [`RELATORIO.md`](RELATORIO.md). Este README foca no **como
> o código funciona e como executá-lo**.

---

## Sumário

- [Visão geral do pipeline](#visão-geral-do-pipeline)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como executar](#como-executar)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Documentação dos módulos](#documentação-dos-módulos)
  - [`dados.py`](#dadospy)
  - [`embeddings.py`](#embeddingspy)
  - [`main.py`](#mainpy)
- [Parâmetros de configuração](#parâmetros-de-configuração)
- [Saídas geradas](#saídas-geradas)
- [Decisões técnicas](#decisões-técnicas)
- [Solução de problemas](#solução-de-problemas)

---

## Visão geral do pipeline

```
frases (texto)
      │
      ▼
[ embeddings.py ]  modelo Transformer all-MiniLM-L6-v2
      │            → matriz 80 × 384
      ▼
[ main.py ] normalização L2
      │
      ▼
   PCA (sklearn.decomposition.PCA)
      │  ├─ análise da variância explicada (todas as componentes)
      │  └─ projeção reduzida a 2 componentes p/ clustering
      ▼
   K-Means para k = 2..10
      │  └─ silhouette_score em cada k → escolhe o maior
      ▼
   K-Means final com o melhor k
      │
      ▼
   análise dos grupos + figuras + CSV + resumo
```

---

## Requisitos

- **Python 3.9+** (testado em 3.13)
- Dependências (em `requirements.txt`):
  - `numpy`, `pandas` — manipulação numérica e tabular
  - `scikit-learn` — PCA, K-Means, silhouette_score
  - `matplotlib`, `seaborn` — gráficos
  - `sentence-transformers` — modelo de deep learning (traz `torch`)

Na primeira execução o modelo `all-MiniLM-L6-v2` (~90 MB) é **baixado
automaticamente** da internet e fica em cache local. As execuções seguintes
são offline.

---

## Instalação

```powershell
pip install -r requirements.txt
```

---

## Como executar

```powershell
python main.py
```

O script imprime o progresso no terminal e grava os artefatos em `figuras/` e
`resultados/`. Não abre janelas: os gráficos são salvos direto em PNG (backend
`Agg` do matplotlib).

Também é possível rodar cada módulo isoladamente para inspeção:

```powershell
python dados.py        # mostra a contagem de frases por categoria
python embeddings.py   # gera e imprime um embedding de exemplo
```

---

## Estrutura do projeto

```
.
├── dados.py            # dataset: 80 frases rotuladas em 4 categorias
├── embeddings.py       # gera embeddings 384-dim com all-MiniLM-L6-v2
├── main.py             # pipeline PCA → K-Means → silhouette → análise
├── requirements.txt    # dependências
├── README.md           # esta documentação técnica
├── RELATORIO.md        # relatório da atividade (metodologia e resultados)
├── figuras/            # gráficos PNG gerados
│   ├── pca_variancia_explicada.png
│   ├── pca_projecao_2d_categorias.png
│   ├── silhouette_por_k.png
│   └── clusters_kmeans_2d.png
└── resultados/
    ├── atribuicoes.csv # frase, categoria real e cluster atribuído
    └── resumo.txt      # resumo numérico da execução
```

---

## Documentação dos módulos

### `dados.py`

Conjunto de dados de texto usado na atividade.

| Item | Descrição |
|---|---|
| `FRASES` | Lista de tuplas `(frase, categoria)`. São 80 frases, 20 por categoria: `tecnologia`, `esportes`, `culinaria`, `natureza`. |
| `carregar_dados()` | Retorna `(textos, rotulos)` — duas listas paralelas: os textos e suas categorias reais. |

Os rótulos **não** entram no K-Means (que é não supervisionado); servem só para
avaliar, ao final, se os grupos descobertos correspondem aos temas reais.

```python
from dados import carregar_dados
textos, rotulos = carregar_dados()   # 80 textos, 80 rótulos
```

### `embeddings.py`

Converte texto em vetores densos usando um modelo de deep learning.

| Item | Descrição |
|---|---|
| `MODELO_PADRAO` | Nome do modelo: `"all-MiniLM-L6-v2"`. |
| `gerar_embeddings(textos, nome_modelo=MODELO_PADRAO)` | Carrega o modelo, codifica os textos e retorna `(emb, info)`. |

Retorno de `gerar_embeddings`:
- `emb`: `np.ndarray` de forma `(n_textos, 384)`, `dtype=float64`.
- `info`: `dict` com `{"modelo": <nome>, "dimensao": 384}`.

```python
from embeddings import gerar_embeddings
emb, info = gerar_embeddings(textos)   # emb.shape == (80, 384)
```

O módulo também define variáveis de ambiente (`HF_HUB_DISABLE_SYMLINKS_WARNING`,
`TOKENIZERS_PARALLELISM`) e filtra warnings para manter a saída limpa no
Windows.

### `main.py`

Orquestra todo o pipeline. Funções principais:

| Função | Responsabilidade |
|---|---|
| `preparar_pastas()` | Cria `figuras/` e `resultados/` se não existirem. |
| `aplicar_pca(X, linhas_relatorio)` | Normaliza (L2), ajusta o PCA, calcula variância explicada, salva o gráfico de variância acumulada. Retorna `(X_pca_full, pca, X_norm)`. |
| `plot_projecao_categorias(X_pca_full, rotulos)` | Gráfico 2D (PC1×PC2) colorido pelas categorias reais. |
| `escolher_k(X, linhas_relatorio, k_min=2, k_max=10)` | Roda K-Means para cada `k`, calcula `silhouette_score`, escolhe o `k` de maior silhueta e salva o gráfico silhueta×k. Retorna `(melhor_k, scores)`. |
| `aplicar_kmeans_final(X, k)` | Ajusta o K-Means final com o melhor `k`. Retorna `(labels, km)`. |
| `plot_clusters_2d(X_pca_full, labels, k, centroides)` | Gráfico 2D colorido pelos clusters, com os centróides. |
| `analisar_grupos(textos, rotulos, labels, linhas_relatorio)` | Para cada cluster calcula categoria dominante, pureza e composição; calcula o Adjusted Rand Index; salva `atribuicoes.csv`. |
| `main()` | Encadeia tudo e grava `resumo.txt`. |

---

## Parâmetros de configuração

Todos no topo/corpo de `main.py`:

| Parâmetro | Valor padrão | Efeito |
|---|---|---|
| `RANDOM_STATE` | `42` | Semente para PCA e K-Means (reprodutibilidade). |
| `N_COMP_CLUSTER` | `2` | Nº de componentes principais usadas no K-Means. |
| `k_min`, `k_max` (em `escolher_k`) | `2`, `10` | Intervalo de `k` testado pela silhueta. |
| `n_init` (K-Means) | `10` | Reinicializações do K-Means (pega a melhor). |
| `MODELO_PADRAO` (em `embeddings.py`) | `all-MiniLM-L6-v2` | Modelo de deep learning usado. |

Para experimentar: aumente `N_COMP_CLUSTER` (mais informação semântica, silhueta
tende a cair) ou troque `MODELO_PADRAO` por outro modelo de sentence-transformers.

---

## Saídas geradas

**`figuras/`**
- `pca_variancia_explicada.png` — variância acumulada vs. nº de componentes (linhas de 80% e 90%).
- `pca_projecao_2d_categorias.png` — dados em 2D coloridos pela categoria real.
- `silhouette_por_k.png` — silhouette_score para cada `k`, destacando o melhor.
- `clusters_kmeans_2d.png` — clusters do K-Means em 2D, com centróides.

**`resultados/`**
- `atribuicoes.csv` — colunas `texto`, `categoria_real`, `cluster` (UTF-8 com BOM, abre certo no Excel).
- `resumo.txt` — variância do PCA, silhueta por `k`, melhor `k`, composição dos clusters e ARI.

---

## Decisões técnicas

- **Normalização L2 em vez de StandardScaler.** Embeddings de sentence-transformers
  são vetores densos comparados por similaridade de cosseno; normalizar pela norma
  L2 é o pré-processamento recomendado. O `StandardScaler` piorou a recuperação
  dos grupos nos testes.
- **Clustering em 2 componentes.** Rodar K-Means/silhueta nas 384 dimensões (ou em
  dezenas de componentes) sofre com a *maldição da dimensionalidade*: as distâncias
  euclidianas ficam parecidas e o `silhouette_score` cai para perto de zero. Com 2
  componentes a silhueta fica bem definida e ainda apontou `k = 4`, que coincide
  com o número real de categorias.
- **`silhouette_score` como critério de `k`.** Métrica não supervisionada que mede
  o quão bem cada ponto se encaixa no próprio cluster versus o cluster vizinho;
  escolhemos o `k` de maior valor.
- **Backend `Agg` do matplotlib.** Salva PNGs sem abrir janela, adequado para
  execução em terminal/CI.

---

## Solução de problemas

- **Aviso de *symlink* do Hugging Face no Windows.** Inofensivo (só afeta a
  eficiência do cache). Já é silenciado via `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- **A primeira execução demora.** É o download único do modelo (~90 MB). Depois
  fica em cache (`~/.cache/huggingface`).
- **`exit code 1` no PowerShell mesmo terminando OK.** O PowerShell trata
  qualquer texto em *stderr* (como warnings) como erro. O script conclui
  normalmente; confira a mensagem final `Concluido.` e os arquivos em
  `resultados/`.
- **Sem internet na primeira execução.** O download do modelo falhará. Rode uma
  vez com conexão para popular o cache.
```
