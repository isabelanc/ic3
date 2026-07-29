"""Plota o mapa de facies (grupos K-means) em uma fatia de tempo/profundidade
fixa (Z-slice), no plano inline x crossline. Pressupoe que 'df_com_pca' (com
a coluna 'grupo_kmeans'), 'mapa_facies' e 'k_usado' ja existem no namespace
(gerados por clusterizar_pcs.py / plotar_facies_inline.py)."""

import matplotlib.pyplot as plt

indice_z_alvo = 75

# filtra o dataframe usando o novo índice
df_zslice = df_com_pca[df_com_pca["sample_idx"] == indice_z_alvo].copy()
df_pivot_z = df_zslice.pivot(index="inline", columns="xline", values="grupo_kmeans")

inlines_mapa = df_pivot_z.index.to_numpy()
xlines_mapa = df_pivot_z.columns.to_numpy()
# extent + origin='lower': mapa em planta, sem eixo de tempo -- ao contrario
# das secoes verticais (onde tempo cresce pra baixo por convencao sismica),
# aqui inline deve crescer para cima, como em qualquer mapa cartesiano
extensao = [xlines_mapa.min(), xlines_mapa.max(), inlines_mapa.min(), inlines_mapa.max()]

plt.figure(figsize=(8, 6))

# vmin=0 e vmax=k_usado-1 garantem que a paleta de cores não quebre caso a matriz não tenha todos os grupos K-means de 0 a 4.

im_z = plt.imshow(
    df_pivot_z.values, cmap=mapa_facies, aspect="auto",
    vmin=0, vmax=k_usado - 1, extent=extensao, origin="lower",
)

plt.title(f"Fácies Sísmicas via K-means - Índice da Amostra: {indice_z_alvo}", fontsize=14)
plt.xlabel("Crossline", fontsize=12)
plt.ylabel("Inline", fontsize=12)

plt.colorbar(im_z, ticks=range(k_usado), label="Grupo (Fácies K-means)")

plt.tight_layout()
plt.show()
