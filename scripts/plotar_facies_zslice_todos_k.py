"""Plota o mapa de facies (grupos K-means) em uma fatia de tempo/profundidade
fixa (Z-slice), no plano inline x crossline, um subplot para cada K testado
(3 a 8). Pressupoe que 'df_com_pca', 'df_grupos_por_k' e 'centroides_por_k'
ja existem no namespace (gerados por clusterizar_pcs.py) -- reaproveita os
rotulos ja calculados la para todos os K, sem re-treinar."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

# 8 cores para cobrir ate K=8 (faixa_k = range(3, 9) em clusterizar_pcs.py)
cores_base = [
    "black", "dodgerblue", "forestgreen", "gold", "lightgray",
    "darkorange", "purple", "crimson",
]

indice_z_alvo = 75

mask_zslice = df_com_pca["sample_idx"] == indice_z_alvo
df_geo_zslice = df_com_pca.loc[mask_zslice, ["inline", "xline"]]
df_grupos_zslice = df_grupos_por_k.loc[mask_zslice]

faixa_k = sorted(centroides_por_k.keys())

fig, axes = plt.subplots(
    nrows=len(faixa_k), ncols=1, figsize=(8, 6 * len(faixa_k)), squeeze=False,
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

    df_pivot_z = df_geo_zslice.assign(
        grupo=df_grupos_zslice[f"grupo_k{k}"]
    ).pivot(index="inline", columns="xline", values="grupo")

    inlines_mapa = df_pivot_z.index.to_numpy()
    xlines_mapa = df_pivot_z.columns.to_numpy()
    # extent + origin='lower': mapa em planta, sem eixo de tempo -- ao
    # contrario das secoes verticais (onde tempo cresce pra baixo por
    # convencao sismica), aqui inline deve crescer para cima, como em
    # qualquer mapa cartesiano
    extensao = [xlines_mapa.min(), xlines_mapa.max(), inlines_mapa.min(), inlines_mapa.max()]

    im = eixo_atual.imshow(
        df_pivot_z.values, cmap=mapa_facies, norm=norma_facies,
        aspect="auto", extent=extensao, origin="lower",
    )

    eixo_atual.set_title(f"Fácies Sísmicas via K-means (K={k}) - Índice da Amostra: {indice_z_alvo}", fontsize=14)
    eixo_atual.set_xlabel("Crossline", fontsize=12)
    eixo_atual.set_ylabel("Inline", fontsize=12)

    fig.colorbar(im, ax=eixo_atual, ticks=range(k_usado), label="Grupo (Fácies K-means)")

plt.tight_layout()
plt.show()
