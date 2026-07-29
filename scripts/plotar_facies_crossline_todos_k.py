"""Plota o mapa de facies (grupos K-means) de uma crossline especifica, um
subplot para cada K testado (3 a 8). Pressupoe que 'df_com_pca',
'df_grupos_por_k' e 'centroides_por_k' ja existem no namespace (gerados por
clusterizar_pcs.py) -- reaproveita os rotulos ja calculados la para todos
os K, sem re-treinar."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

# 8 cores para cobrir ate K=8 (faixa_k = range(3, 9) em clusterizar_pcs.py)
cores_base = [
    "black", "dodgerblue", "forestgreen", "gold", "lightgray",
    "darkorange", "purple", "crimson",
]

xline_alvo = 9558

mask_crossline = df_com_pca["xline"] == xline_alvo
df_geo_crossline = df_com_pca.loc[mask_crossline, ["inline", "sample_idx"]]
df_grupos_crossline = df_grupos_por_k.loc[mask_crossline]

faixa_k = sorted(centroides_por_k.keys())

fig, axes = plt.subplots(
    nrows=len(faixa_k), ncols=1, figsize=(12, 4 * len(faixa_k)), squeeze=False,
)
axes = axes.ravel()

for eixo_atual, k in zip(axes, faixa_k):
    k_usado = len(centroides_por_k[k])
    assert k_usado <= len(cores_base), (
        f"cores_base tem {len(cores_base)} cores, mas k_usado={k_usado}. "
        "Adicione mais cores em cores_base."
    )

    mapa_facies = ListedColormap(cores_base[:k_usado])

    # limites entre os grupos em n - 0.5, garantindo que o valor inteiro n
    # caia sempre na n-esima cor, independente de quais grupos aparecem na
    # fatia plotada
    limites_grupos = np.arange(k_usado + 1) - 0.5
    norma_facies = BoundaryNorm(limites_grupos, mapa_facies.N)

    df_pivot_crossline = df_geo_crossline.assign(
        grupo=df_grupos_crossline[f"grupo_k{k}"]
    ).pivot(index="sample_idx", columns="inline", values="grupo")

    inlines_secao = df_pivot_crossline.columns.to_numpy()
    extensao = [inlines_secao.min(), inlines_secao.max(), df_pivot_crossline.shape[0] - 1, 0]

    im = eixo_atual.imshow(
        df_pivot_crossline.values, cmap=mapa_facies, norm=norma_facies,
        aspect="auto", extent=extensao,
    )

    eixo_atual.set_title(f"Fácies Sísmicas via K-means (K={k}) - Crossline {xline_alvo}", fontsize=14)
    eixo_atual.set_ylabel("Amostras / Tempo", fontsize=12)

    fig.colorbar(im, ax=eixo_atual, ticks=range(k_usado), label="Grupo (Fácies)")

axes[-1].set_xlabel("Inline", fontsize=12)

plt.tight_layout()
plt.show()
