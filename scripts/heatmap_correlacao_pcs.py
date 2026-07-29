"""Heatmap de correlacao entre atributos sismicos originais e componentes
principais. Pressupoe que 'df_com_pca' e 'colunas_atributos' ja existem no
namespace (gerados por aplicar_pca.py)."""

import matplotlib.pyplot as plt
import seaborn as sns

colunas_pcs = ["PC1", "PC2", "PC3", "PC4", "PC5"]

# colunas_pcs = df_pca.columns.tolist()

# Calcula a matriz de correlação
matriz_correlacao_completa = df_com_pca[colunas_atributos + colunas_pcs].corr()

# Filtra apenas o cruzamento Atributos x PCs
correlacao_focada = matriz_correlacao_completa.loc[colunas_atributos, colunas_pcs]

correlacao_focada = correlacao_focada.T

correlacao_focada = correlacao_focada.rename(
    index={
        "PC1": "1ª Componente",
        "PC2": "2ª Componente",
        "PC3": "3ª Componente",
        "PC4": "4ª Componente",
        "PC5": "5ª Componente",
    },
    columns={
        "sweetness": "Sweetness",
        "sd_15hz": "Decomposição Espectral (15 Hz)",
        "sd_20hz": "Decomposição Espectral (20 Hz)",
        "sd_25hz": "Decomposição Espectral (25 Hz)",
        "sd_30hz": "Decomposição Espectral (30 Hz)",
        "sd_35hz": "Decomposição Espectral (35 Hz)",
        "sd_40hz": "Decomposição Espectral (40 Hz)",
        "sd_45hz": "Decomposição Espectral (45 Hz)",
        "sd_50hz": "Decomposição Espectral (50 Hz)",
        "amp_inst": "Amplitude Instantânea",
        "amp_rms": "Amplitude RMS",
    },
)

plt.figure(figsize=(12, 6))

sns.heatmap(
    correlacao_focada,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    cbar_kws={"label": "Coeficiente de Pearson"},
)

plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(rotation=45, va="center", fontsize=11)

plt.tight_layout()
plt.show()
