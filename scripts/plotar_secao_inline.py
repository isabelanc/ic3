"""Plota uma secao (inline) de todos os atributos sismicos, reaproveitando a
mascara de borda usada no carregamento do cubo completo (ver
carregar_atributos_sismicos.py)."""

import matplotlib.pyplot as plt
import numpy as np
import segyio as sio

from carregar_atributos_sismicos import (
    atributos_sismicos,
    marcar_bordas_nao_geradas_como_nan,
    pasta_dos_arquivos,
)

numero_inline_desejado = 1326


def ler_secao_inline(caminho_arquivo, numero_inline_desejado: int):
    """Le apenas os tracos de um inline e aplica a mesma mascara de borda
    usada no carregamento do cubo completo."""
    with sio.open(str(caminho_arquivo), "r", ignore_geometry=True) as arquivo_sismico:
        arquivo_sismico.mmap()
        inlines = np.array(arquivo_sismico.attributes(sio.TraceField.INLINE_3D), dtype=np.int32)
        xlines = np.array(arquivo_sismico.attributes(sio.TraceField.CROSSLINE_3D), dtype=np.int32)
        n_samples = len(arquivo_sismico.samples)

        indices_inline = np.where(inlines == numero_inline_desejado)[0]
        if indices_inline.size == 0:
            raise ValueError(
                f"Inline {numero_inline_desejado} nao encontrado em '{caminho_arquivo.name}'."
            )

        secao = np.array(
            [arquivo_sismico.trace[int(tr)] for tr in indices_inline], dtype=np.float32
        )

    secao = marcar_bordas_nao_geradas_como_nan(
        secao.ravel(), n_traces=indices_inline.size, n_samples=n_samples
    ).reshape(indices_inline.size, n_samples)

    return secao, xlines[indices_inline]


def eh_atributo_com_sinal(secao_2d: np.ndarray) -> bool:
    """Atributos de magnitude/energia (decomposicao espectral, envelope) sao
    sempre >= 0; atributos de amplitude/fase podem ser negativos. Detecta
    automaticamente em vez de assumir por nome do atributo."""
    return np.nanmin(secao_2d) < 0


secoes_por_atributo = {}
xlines_por_atributo = {}
for nome_atributo, nome_arquivo in atributos_sismicos.items():
    caminho_arquivo = pasta_dos_arquivos / nome_arquivo
    secao, xlines_secao = ler_secao_inline(caminho_arquivo, numero_inline_desejado)
    secoes_por_atributo[nome_atributo] = secao
    xlines_por_atributo[nome_atributo] = xlines_secao

figura, eixos = plt.subplots(
    nrows=len(atributos_sismicos), ncols=1,
    figsize=(12, 4 * len(atributos_sismicos)), squeeze=False,
)
eixos = eixos.ravel()

for eixo_atual, (nome_atributo, secao_2d) in zip(eixos, secoes_por_atributo.items()):
    xlines_secao = xlines_por_atributo[nome_atributo]
    # extent assume xlines regularmente espacadas dentro do inline (padrao em
    # levantamentos 3D); origem no canto superior para tempo crescer para baixo
    extensao = [xlines_secao.min(), xlines_secao.max(), secao_2d.shape[1] - 1, 0]

    if eh_atributo_com_sinal(secao_2d):
        limite = np.nanmax(np.abs(secao_2d))
        imagem = eixo_atual.imshow(
            secao_2d.T, cmap="seismic", aspect="auto", extent=extensao,
            vmin=-limite, vmax=limite,
        )
    else:
        imagem = eixo_atual.imshow(
            secao_2d.T, cmap="viridis", aspect="auto", extent=extensao, vmin=0,
        )

    eixo_atual.set_title(f"Seção Sísmica - Inline {numero_inline_desejado} ({nome_atributo.upper()})")
    eixo_atual.set_ylabel("Amostras")
    figura.colorbar(imagem, ax=eixo_atual, label="Magnitude")

eixos[-1].set_xlabel("Crossline")
plt.tight_layout()
plt.show()
plt.close(figura)
