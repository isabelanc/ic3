"""Plota o mapa de facies (grupos K-means) de uma crossline especifica, com
cor fixa por grupo (independente de quais grupos aparecem na fatia).
Pressupoe que 'df_com_pca' (com a coluna 'grupo_kmeans') e 'centroides'
ja existem no namespace (gerados por clusterizar_pcs.py)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cores_base = ["black", "dodgerblue", "forestgreen", "gold", "lightgray"]

k_usado = len(centroides)
assert k_usado <= len(cores_base), (
    f"cores_base tem {len(cores_base)} cores, mas k_usado={k_usado}. "
    "Adicione mais cores em cores_base."
)

mapa_facies = ListedColormap(cores_base[:k_usado])

xline_alvo = 9558

df_crossline = df_com_pca[df_com_pca["xline"] == xline_alvo].copy()

df_pivot_crossline = df_crossline.pivot(index="sample_idx", columns="inline", values="grupo_kmeans")
inlines_secao = df_pivot_crossline.columns.to_numpy()
extensao = [inlines_secao.min(), inlines_secao.max(), df_pivot_crossline.shape[0] - 1, 0]

# 3. Plotagem
plt.figure(figsize=(12, 6))

im_crossline = plt.imshow(
    df_pivot_crossline.values, cmap=mapa_facies, aspect="auto",
    vmin=0, vmax=k_usado - 1, extent=extensao,
)

plt.title(f"Fácies Sísmicas via K-means - Crossline {xline_alvo}", fontsize=14)
plt.xlabel("Inline", fontsize=12)  # O eixo X agora representa as Inlines
plt.ylabel("Amostras / Tempo", fontsize=12)

# Adiciona a barra de cores indicando os grupos (de 0 a k-1)
plt.colorbar(im_crossline, ticks=range(k_usado), label="Grupo (Fácies)")

plt.tight_layout()
plt.show()
