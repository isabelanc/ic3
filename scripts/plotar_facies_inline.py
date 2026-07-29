"""Plota o mapa de facies (grupos K-means) de um inline especifico, com
cor fixa por grupo (independente de quais grupos aparecem na fatia).
Pressupoe que 'df_com_pca' (com a coluna 'grupo_kmeans') e 'centroides'
ja existem no namespace (gerados por clusterizar_pcs.py)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

cores_base = ["black", "dodgerblue", "forestgreen", "gold", "lightgray"]

k_usado = len(centroides)
assert k_usado <= len(cores_base), (
    f"cores_base tem {len(cores_base)} cores, mas k_usado={k_usado}. "
    "Adicione mais cores em cores_base."
)

mapa_facies = ListedColormap(cores_base[:k_usado])

# limites entre os grupos em n - 0.5, garantindo que o valor inteiro n
# caia sempre na n-esima cor, independente de quais grupos aparecem na
# fatia plotada (sem isso, o imshow normaliza pelo min/max PRESENTES na
# fatia e as cores nao correspondem mais ao mesmo grupo entre secoes)
limites_grupos = np.arange(k_usado + 1) - 0.5
norma_facies = BoundaryNorm(limites_grupos, mapa_facies.N)

inline_alvo = 1326

df_inline = df_com_pca[df_com_pca["inline"] == inline_alvo].copy()

df_pivot_inline = df_inline.pivot(index="sample_idx", columns="xline", values="grupo_kmeans")
xlines_secao = df_pivot_inline.columns.to_numpy()
extensao = [xlines_secao.min(), xlines_secao.max(), df_pivot_inline.shape[0] - 1, 0]

plt.figure(figsize=(12, 6))

im_inline = plt.imshow(
    df_pivot_inline.values, cmap=mapa_facies, norm=norma_facies,
    aspect="auto", extent=extensao,
)

plt.title(f"Fácies Sísmicas via K-means - Inline {inline_alvo}", fontsize=14)
plt.xlabel("Crossline", fontsize=12)
plt.ylabel("Amostras / Tempo", fontsize=12)

# Adiciona a barra de cores indicando os grupos (de 0 a k-1)
plt.colorbar(im_inline, ticks=range(k_usado), label="Grupo (Fácies)")

plt.tight_layout()
plt.show()
