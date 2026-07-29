"""Plota, para um inline especifico, a secao 2D de cada componente
principal (PC). Pressupoe que 'df_com_pca' ja existe no namespace (gerado
por aplicar_pca.py)."""

import matplotlib.pyplot as plt
import numpy as np

inline_alvo = 1326

pcs_para_plotar = ["PC1", "PC2", "PC3", "PC4", "PC5"]

# satura (preto/branco) os valores acima do percentil abaixo, aumentando o
# contraste do restante dos dados; abaixe para mais contraste, suba para
# mais fidelidade aos extremos
percentil_clip = 98

df_inline = df_com_pca[df_com_pca["inline"] == inline_alvo].copy()

fig, axes = plt.subplots(
    nrows=len(pcs_para_plotar), ncols=1, figsize=(12, 4 * len(pcs_para_plotar))
)

for ax, pc in zip(axes, pcs_para_plotar):
    df_pivot = df_inline.pivot(index="sample_idx", columns="xline", values=pc)
    matriz_2d = df_pivot.values
    xlines_secao = df_pivot.columns.to_numpy()

    # PCs sao centrados em zero pelo PCA -> escala simetrica em torno de
    # zero, como ja usado nos outros plots de secao sismica. Usa percentil
    # em vez do maximo absoluto para nao deixar outliers achatarem o
    # contraste do restante dos dados.
    limite = np.nanpercentile(np.abs(matriz_2d), percentil_clip)
    extensao = [xlines_secao.min(), xlines_secao.max(), matriz_2d.shape[0] - 1, 0]

    im = ax.imshow(
        matriz_2d, cmap="grey", aspect="auto", extent=extensao,
        vmin=-limite, vmax=limite,
    )

    ax.set_title(f"Seção Sísmica - {pc} (Inline {inline_alvo})")
    ax.set_ylabel("Amostras (Tempo)")

    fig.colorbar(im, ax=ax, label=f"Valor do {pc}")

axes[-1].set_xlabel("Crossline")

plt.tight_layout()
plt.show()
