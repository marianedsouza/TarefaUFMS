# Embeddings de Texto em Português + PCA + K-Means

Documentação técnica do projeto. Ele representa 150 textos em português como
vetores (*embeddings*) usando um modelo de deep learning multilíngue, reduz a
dimensionalidade com **PCA (2D e 3D)** e descobre grupos com **K-Means**,
escolhendo o número de grupos pelo **método do cotovelo** e pelo
**silhouette_score**. Segue a prática da aula de Ciência de Dados.

> Relatório da atividade (formato de entrega): [`RELATORIO.md`](RELATORIO.md).
> Este README foca em **como o código funciona e como executá-lo**.

---

## Visão geral do pipeline

```
dataset_150_textos_portugues.csv  (150 textos, 6 categorias)
      │
      ▼
[ embeddings.py ]  paraphrase-multilingual-MiniLM-L12-v2  → 150 × 384
      │
      ▼
[ main.py ]  normalização L2
      │
      ├─ PCA (2D e 3D)  → visualização
      └─ K-Means no espaço completo (384-d)
             ├─ método do cotovelo (inércia × k)
             └─ silhouette_score × k  → escolhe o melhor k
      │
      ▼
  caracterização dos grupos + crosstab + figuras + CSVs
```

O K-Means roda no **espaço completo** dos embeddings (como na aula); o PCA é usado
apenas para **visualizar** os resultados em 2D e 3D.

---

## Requisitos

- **Python 3.9+** (testado em 3.13)
- Dependências (`requirements.txt`): `numpy`, `pandas`, `scikit-learn`,
  `matplotlib`, `seaborn`, `sentence-transformers` (traz `torch`).

Na primeira execução o modelo multilíngue (~470 MB) é baixado do Hugging Face e
fica em cache local.

---

## Instalação e execução

```powershell
pip install -r requirements.txt
python criar_dataset.py   # gera o CSV (opcional: main.py gera se faltar)
python main.py            # pipeline completo
python gerar_html.py      # (opcional) monta index.html com os resultados
```

Para ver o `index.html` no navegador (as imagens precisam de HTTP):

```powershell
python -m http.server 8010
# abra http://localhost:8010/index.html
```

---

## Estrutura do projeto

```
.
├── criar_dataset.py                  # gera a base de 150 textos (6 categorias)
├── dataset_150_textos_portugues.csv  # base reprodutível (texto, categoria)
├── dados.py                          # carrega o CSV
├── embeddings.py                     # embeddings 384-d (modelo multilíngue)
├── main.py                           # PCA 2D/3D + cotovelo + silhouette + K-Means
├── gerar_html.py                     # monta index.html a partir dos resultados
├── index.html                        # página visual dos resultados
├── requirements.txt
├── README.md                         # esta documentação
├── RELATORIO.md                      # relatório da atividade (formato de entrega)
├── figuras/                          # 6 gráficos PNG
│   ├── pca_2d_sem_rotulo.png
│   ├── pca_2d_categorias.png
│   ├── pca_2d_clusters.png
│   ├── pca_3d_clusters.png
│   ├── metodo_cotovelo.png
│   └── silhouette_por_k.png
└── resultados/
    ├── atribuicoes.csv               # texto, categoria real, cluster
    ├── crosstab_categoria_cluster.csv
    └── resumo.txt
```

---

## Documentação dos módulos

### `criar_dataset.py`
Define `CATEGORIAS` (6 temas × 25 frases) e grava
`dataset_150_textos_portugues.csv` com colunas `texto` e `categoria`.

### `dados.py`
| Função | Descrição |
|---|---|
| `carregar_df(caminho)` | Retorna o `DataFrame` (`texto`, `categoria`); gera o CSV se faltar. |
| `carregar_dados(caminho)` | Retorna `(textos, rotulos)` como listas. |

### `embeddings.py`
| Item | Descrição |
|---|---|
| `MODELO_PADRAO` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. |
| `gerar_embeddings(textos, nome_modelo)` | Retorna `(emb, info)`; `emb` é `(n, 384)`. |

### `main.py`
| Função | Responsabilidade |
|---|---|
| `aplicar_pca(X, rel)` | PCA de 3 componentes; variância explicada. Retorna `(X_pca, pca)`. |
| `plot_pca_2d_sem_rotulo` / `plot_pca_2d_categorias` | Projeções 2D (bruta e por categoria). |
| `plot_pca_2d_clusters` / `plot_pca_3d_clusters` | Clusters em 2D e 3D. |
| `escolher_k(X, rel, k_min, k_max)` | Cotovelo + silhouette; escolhe o k de maior silhueta. |
| `analisar_grupos(df, rel)` | Categoria dominante, pureza, exemplos, crosstab e ARI. |
| `main()` | Encadeia dados → embeddings → L2 → PCA → K-Means → análise. |

---

## Parâmetros de configuração

| Parâmetro | Local | Padrão | Efeito |
|---|---|---|---|
| `RANDOM_STATE` | `main.py` | `42` | Reprodutibilidade de PCA e K-Means. |
| `k_min`, `k_max` | `escolher_k` | `2`, `10` | Faixa de k avaliada. |
| `n_init` | K-Means | `10` | Reinicializações do K-Means. |
| `MODELO_PADRAO` | `embeddings.py` | multilíngue MiniLM-L12 | Modelo de embeddings. |
| categorias/frases | `criar_dataset.py` | 6 × 25 | Conteúdo da base. |

---

## Resultado obtido

- `silhouette_score` máximo em **k = 6** = número real de categorias.
- **Adjusted Rand Index ≈ 0,85**; pureza de **92%–96%** por grupo.
- Cada grupo corresponde a um tema: culinária, esportes, tecnologia, educação,
  saúde e natureza. Detalhes e interpretação em `RELATORIO.md`.

---

## Solução de problemas

- **Aviso de symlink do Hugging Face (Windows):** inofensivo; já silenciado.
- **Primeira execução demora:** download único do modelo (~470 MB), depois fica
  em cache (`~/.cache/huggingface`).
- **`exit code 1` no PowerShell mesmo terminando OK:** o PowerShell trata texto
  em *stderr* (warnings) como erro; confira a mensagem final `Concluido.`.
