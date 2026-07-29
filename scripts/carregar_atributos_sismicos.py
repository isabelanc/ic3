"""Carrega multiplos atributos sismicos (SEGY) e monta um DataFrame unico
indexado por (inline, xline, sample_idx)."""

from pathlib import Path

import numpy as np
import pandas as pd
import segyio as sio

pasta_dos_arquivos = Path(".")

atributos_sismicos = {
    "sweetness": "Sweetness.sgy",
    # "similaridade": "similarity.sgy",
    "sd_15hz": "SD_15hz.sgy",
    "sd_20hz": "SD_20hz.sgy",
    "sd_25hz": "25Hz_spectral_decomposition.sgy",
    "sd_30hz": "SD_30hz.sgy",
    "sd_35hz": "SD_35hz.sgy",
    "sd_40hz": "SD_40hz.sgy",
    "sd_45hz": "45Hz_spectral_decomposition.sgy",
    "sd_50hz": "SD_50hz.sgy",
    "amp_inst": "instantaneous_amplitude.sgy",
    "amp_rms": "Amp_RMS.sgy",
}

nomes_atributos = list(atributos_sismicos.keys())


def ler_geometria_e_valores(caminho_arquivo: Path, geometria_referencia: dict | None):
    """Le cabecalho + tracos do SEGY"""
    with sio.open(str(caminho_arquivo), "r", ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount
        n_samples = len(f.samples)
        inlines = np.array(f.attributes(sio.TraceField.INLINE_3D), dtype=np.int32)
        xlines = np.array(f.attributes(sio.TraceField.CROSSLINE_3D), dtype=np.int32)
        sample_time = np.array(f.samples, dtype=np.float32)

        try:
            valores = f.trace.raw[:].ravel()
        except AttributeError:
            valores = np.array([f.trace[tr] for tr in range(n_traces)], dtype=np.float32).ravel()
        valores = valores.astype(np.float32, copy=False)

    if geometria_referencia is None:
        assert n_samples <= np.iinfo(np.int16).max, (
            f"n_samples={n_samples} excede o limite do int16 usado em 'sample_idx'."
        )
        geometria_referencia = {
            "n_traces": n_traces, "n_samples": n_samples,
            "inlines": inlines, "xlines": xlines, "sample_time": sample_time,
        }
    elif (
        n_traces != geometria_referencia["n_traces"]
        or n_samples != geometria_referencia["n_samples"]
        or not np.array_equal(inlines, geometria_referencia["inlines"])
        or not np.array_equal(xlines, geometria_referencia["xlines"])
        or not np.array_equal(sample_time, geometria_referencia["sample_time"])
    ):
        raise ValueError(
            f"'{caminho_arquivo.name}' tem geometria diferente da referencia "
            f"({n_traces} tracos x {n_samples} amostras vs. "
            f"{geometria_referencia['n_traces']} x {geometria_referencia['n_samples']} esperados)."
        )

    return valores, geometria_referencia


def marcar_bordas_nao_geradas_como_nan(
    valores: np.ndarray, n_traces: int, n_samples: int
) -> np.ndarray:
    """Marca como NaN apenas os zeros de topo/base de cada traco (amostras que o
    OpendTect nao gera por efeito de borda da transformada tempo-frequencia),
    preservando zeros genuinos no meio do traco (ex.: cruzamentos de zero)."""
    matriz = valores.reshape(n_traces, n_samples)
    nao_zero = matriz != 0

    tem_dado = nao_zero.any(axis=1)
    primeiro_idx = np.argmax(nao_zero, axis=1)
    ultimo_idx = n_samples - 1 - np.argmax(nao_zero[:, ::-1], axis=1)

    indices_amostra = np.arange(n_samples)
    e_borda = (
        (indices_amostra[None, :] < primeiro_idx[:, None])
        | (indices_amostra[None, :] > ultimo_idx[:, None])
    )
    e_borda |= ~tem_dado[:, None]  # traco inteiramente zero -> sem dado gerado

    matriz = matriz.copy()
    matriz[e_borda] = np.nan
    return matriz.ravel()


if __name__ == "__main__":
    colunas_atributos = {}

    geometria_referencia = None

    for nome_atributo in nomes_atributos:
        caminho_arquivo = pasta_dos_arquivos / atributos_sismicos[nome_atributo]
        if not caminho_arquivo.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {caminho_arquivo}")

        print(f"Lendo atributo: {nome_atributo}")
        valores, geometria_referencia = ler_geometria_e_valores(caminho_arquivo, geometria_referencia)

        valores = marcar_bordas_nao_geradas_como_nan(
            valores, geometria_referencia["n_traces"], geometria_referencia["n_samples"]
        )
        colunas_atributos[nome_atributo] = valores

    n_traces = geometria_referencia["n_traces"]
    n_samples = geometria_referencia["n_samples"]

    colunas_geometria = {
        "inline": np.repeat(geometria_referencia["inlines"], n_samples),
        "xline": np.repeat(geometria_referencia["xlines"], n_samples),
        "sample_idx": np.tile(np.arange(n_samples, dtype=np.int16), n_traces),
        "sample_time": np.tile(geometria_referencia["sample_time"], n_traces),
    }

    df_final = pd.DataFrame({**colunas_geometria, **colunas_atributos})
    del colunas_geometria, colunas_atributos

    print(f"\nDataFrame final: {len(df_final):,} linhas")
    print(f"Memoria: {df_final.memory_usage(deep=True).sum() / 2**30:.2f} GB")
    print(df_final.head())
