"""Normaliza (z-score) os atributos sismicos do DataFrame montado por
carregar_atributos_sismicos.py. Pressupoe que 'df_final' ja existe no
namespace (gerado ao rodar o loader antes deste script/celula)."""

import pandas as pd

colunas_atributos = [
    "sd_10hz", "sd_15hz", "sd_20hz", "sd_25hz", "sd_30hz", "sd_35hz",
    "sd_40hz", "sd_45hz", "sd_50hz", "sd_55hz", "amp_inst",
]


def normalizacao(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Normalizacao z-score (media 0, desvio 1) por coluna, preservando os
    NaN de amostras invalidas (bordas nao geradas)."""
    df_norm = df.copy()

    for col in colunas:
        media = df_norm[col].mean()
        desvio = df_norm[col].std(ddof=0)  # divide pelo numero total de amostras (N)

        if desvio == 0 or pd.isna(desvio):
            # coluna constante ou sem nenhum valor valido: nao ha o que
            # normalizar por escala; zera so as amostras validas e preserva
            # NaN nas amostras invalidas (bordas)
            df_norm[col] = df_norm[col].where(df_norm[col].isna(), 0.0)
        else:
            df_norm[col] = (df_norm[col] - media) / desvio

    return df_norm


df_normalizado = normalizacao(df_final, colunas_atributos)
