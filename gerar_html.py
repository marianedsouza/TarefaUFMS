"""
Gera uma pagina HTML (index.html) reunindo os resultados da atividade:
figuras do PCA/K-Means, tabela de silhueta, analise dos grupos e a tabela
completa de atribuicoes. Use depois de rodar main.py.

    python gerar_html.py
"""

import csv
import os
from collections import Counter, defaultdict

PASTA_RES = "resultados"
CSV = os.path.join(PASTA_RES, "atribuicoes.csv")
SAIDA = "index.html"


def ler_atribuicoes():
    linhas = []
    with open(CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["cluster"] = int(row["cluster"])
            linhas.append(row)
    return linhas


def ler_resumo():
    caminho = os.path.join(PASTA_RES, "resumo.txt")
    if os.path.exists(caminho):
        with open(caminho, encoding="utf-8") as f:
            return f.read()
    return ""


def tabela_grupos(linhas):
    """Monta os dados de composicao/pureza por cluster."""
    por_cluster = defaultdict(list)
    for r in linhas:
        por_cluster[r["cluster"]].append(r)

    blocos = []
    for c in sorted(por_cluster):
        frases = por_cluster[c]
        contagem = Counter(f["categoria_real"] for f in frases)
        dominante, qtd = contagem.most_common(1)[0]
        pureza = qtd / len(frases)
        composicao = ", ".join(f"{k}: {v}" for k, v in contagem.most_common())
        exemplos = [f["texto"] for f in frases[:3]]
        blocos.append({
            "cluster": c,
            "n": len(frases),
            "dominante": dominante,
            "pureza": pureza,
            "composicao": composicao,
            "exemplos": exemplos,
        })
    return blocos


def esc(texto):
    return (texto.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;"))


def gerar():
    linhas = ler_atribuicoes()
    blocos = tabela_grupos(linhas)
    resumo = ler_resumo()

    cores = ["#2563eb", "#16a34a", "#ea580c", "#9333ea",
             "#dc2626", "#0891b2", "#ca8a04", "#db2777"]

    cards = ""
    for b in blocos:
        cor = cores[b["cluster"] % len(cores)]
        exemplos = "".join(f"<li>{esc(e)}</li>" for e in b["exemplos"])
        cards += f"""
        <div class="card" style="border-top:4px solid {cor}">
          <h3>Cluster {b['cluster']}</h3>
          <p class="dom">Categoria dominante: <b>{b['dominante']}</b></p>
          <p class="metric">{b['n']} frases &middot; pureza {b['pureza']:.0%}</p>
          <p class="comp">{esc(b['composicao'])}</p>
          <p class="lbl">Exemplos:</p>
          <ul>{exemplos}</ul>
        </div>"""

    linhas_tabela = ""
    for r in linhas:
        cor = cores[r["cluster"] % len(cores)]
        linhas_tabela += (
            f"<tr><td>{esc(r['texto'])}</td>"
            f"<td>{esc(r['categoria_real'])}</td>"
            f"<td><span class='chip' style='background:{cor}'>"
            f"{r['cluster']}</span></td></tr>"
        )

    figuras = [
        ("figuras/pca_variancia_explicada.png",
         "PCA - Variancia explicada acumulada"),
        ("figuras/pca_projecao_2d_categorias.png",
         "Projecao PCA 2D por categoria real"),
        ("figuras/silhouette_por_k.png",
         "Escolha de k pelo silhouette_score"),
        ("figuras/clusters_kmeans_2d.png",
         "Grupos do K-Means (k=4) em 2D"),
    ]
    figs_html = ""
    for src, titulo in figuras:
        if os.path.exists(src):
            figs_html += f"""
            <figure>
              <img src="{src}" alt="{esc(titulo)}">
              <figcaption>{esc(titulo)}</figcaption>
            </figure>"""

    html = f"""<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PCA + K-Means sobre Embeddings de Deep Learning</title>
<style>
  :root {{ --bg:#0f172a; --panel:#1e293b; --txt:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:system-ui,Segoe UI,Roboto,sans-serif;
          background:var(--bg); color:var(--txt); line-height:1.55; }}
  header {{ padding:40px 24px; text-align:center;
            background:linear-gradient(135deg,#1e3a8a,#0f172a); }}
  header h1 {{ margin:0 0 8px; font-size:1.8rem; }}
  header p {{ margin:0; color:var(--muted); }}
  main {{ max-width:1100px; margin:0 auto; padding:24px; }}
  section {{ margin:36px 0; }}
  h2 {{ border-left:4px solid var(--accent); padding-left:12px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
           gap:20px; }}
  figure {{ margin:0; background:var(--panel); border-radius:12px; padding:12px;
            box-shadow:0 4px 12px rgba(0,0,0,.3); }}
  figure img {{ width:100%; border-radius:8px; background:#fff; }}
  figcaption {{ text-align:center; color:var(--muted); margin-top:8px;
                font-size:.9rem; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
            gap:16px; }}
  .card {{ background:var(--panel); border-radius:12px; padding:16px; }}
  .card h3 {{ margin:0 0 8px; }}
  .card .dom {{ margin:4px 0; }}
  .card .metric {{ color:var(--accent); font-weight:600; margin:4px 0; }}
  .card .comp {{ color:var(--muted); font-size:.9rem; }}
  .card .lbl {{ margin:10px 0 4px; color:var(--muted); font-size:.85rem; }}
  .card ul {{ margin:0; padding-left:18px; font-size:.88rem; }}
  .highlight {{ background:var(--panel); border-radius:12px; padding:20px 24px;
                border-left:4px solid #16a34a; }}
  .highlight b {{ color:#4ade80; font-size:1.15rem; }}
  table {{ width:100%; border-collapse:collapse; background:var(--panel);
           border-radius:12px; overflow:hidden; font-size:.9rem; }}
  th,td {{ text-align:left; padding:10px 12px; border-bottom:1px solid #334155; }}
  th {{ background:#334155; }}
  .chip {{ display:inline-block; min-width:24px; text-align:center; color:#fff;
           padding:2px 8px; border-radius:999px; font-weight:600; }}
  pre {{ background:#0b1220; padding:16px; border-radius:12px; overflow:auto;
         color:#cbd5e1; font-size:.82rem; }}
  footer {{ text-align:center; color:var(--muted); padding:30px; }}
</style>
</head>
<body>
<header>
  <h1>Reducao de Dimensionalidade (PCA) e Agrupamento (K-Means)</h1>
  <p>Embeddings gerados por deep learning &middot; modelo all-MiniLM-L6-v2 (384 dimensoes)</p>
</header>
<main>

  <section class="highlight">
    <p>Numero de grupos escolhido pelo <code>silhouette_score</code>:
       <b>k = 4</b> (silhueta = 0,4170) &mdash; coincidindo com as 4 categorias reais.</p>
  </section>

  <section>
    <h2>Graficos</h2>
    <div class="grid">{figs_html}
    </div>
  </section>

  <section>
    <h2>Grupos encontrados</h2>
    <div class="cards">{cards}
    </div>
  </section>

  <section>
    <h2>Atribuicoes (todas as frases)</h2>
    <table>
      <thead><tr><th>Frase</th><th>Categoria real</th><th>Cluster</th></tr></thead>
      <tbody>{linhas_tabela}</tbody>
    </table>
  </section>

  <section>
    <h2>Resumo numerico da execucao</h2>
    <pre>{esc(resumo)}</pre>
  </section>

</main>
<footer>Gerado automaticamente a partir dos resultados de <code>main.py</code></footer>
</body>
</html>"""

    with open(SAIDA, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Pagina gerada: {os.path.abspath(SAIDA)}")


if __name__ == "__main__":
    gerar()
