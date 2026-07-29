"""Aplica PCA sobre os atributos sismicos normalizados (ver
normalizar_atributos.py). Pressupoe que 'df_normalizado' e 'colunas_atributos'
ja existem no namespace (gerados pelas etapas anteriores)."""

import pandas as pd
from sklearn.decomposition import PCA


def aplicar_pca(df: pd.DataFrame, colunas: list[str], n_componentes: int | None = None):
    df = df.copy()

    df_atributos = df[colunas]

    mask_valida = df_atributos.notna().all(axis=1)  # linhas com todos os atributos validos

    df_valido = df_atributos[mask_valida]

    pca = PCA(n_components=n_componentes)
    valores_pca = pca.fit_transform(df_valido)

    nomes_pcs = [f"PC{i + 1}" for i in range(valores_pca.shape[1])]

    df_pca = pd.DataFrame(valores_pca, columns=nomes_pcs, index=df_valido.index)  # mesmo numero de linhas

    df_pca_completo = df_pca.reindex(df.index)

    df[nomes_pcs] = df_pca_completo
    print(f"Linhas originais: {len(df):,}")
    print(f"Linhas usadas no PCA: {len(df_valido):,}")
    print("Variancia explicada por componente:")
    for nome, variancia in zip(nomes_pcs, pca.explained_variance_ratio_):
        print(f"  {nome}: {variancia:.2%}")
    print(f"  Total: {pca.explained_variance_ratio_.sum():.2%}")

    return df, pca, df_pca_completo


df_com_pca, modelo_pca, df_pca = aplicar_pca(df_normalizado, colunas_atributos)
