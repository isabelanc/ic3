"""Plota o scree plot do PCA (variancia individual e acumulada por
componente). Pressupoe que 'modelo_pca' ja existe no namespace (gerado por
aplicar_pca.py)."""

import matplotlib.pyplot as plt
import numpy as np

var_fracao = modelo_pca.explained_variance_ratio_
var_fracao_perc = var_fracao * 100
var_acumulada_perc = np.cumsum(var_fracao_perc)
numero_componentes = np.arange(1, len(var_fracao) + 1)

for i, v in enumerate(var_fracao, start=1):
    print(f"O PC{i} explica {v * 100:.2f}% dos dados.")
print(f"Total de informação preservada: {var_fracao.sum() * 100:.2f}%")

figura, eixo_individual = plt.subplots(figsize=(10, 5))

eixo_individual.bar(
    numero_componentes, var_fracao_perc, color="navy", alpha=0.6, label="Variância individual"
)
eixo_individual.set_xlabel("Número de componentes principais", fontsize=12)
eixo_individual.set_ylabel("Variância individual explicada (%)", fontsize=12, color="navy")
eixo_individual.tick_params(axis="y", labelcolor="navy")
eixo_individual.set_xticks(numero_componentes)
eixo_individual.grid(True, alpha=0.3)

eixo_acumulado = eixo_individual.twinx()
eixo_acumulado.plot(
    numero_componentes, var_acumulada_perc, marker="o", color="firebrick", linestyle="-",
    label="Variância acumulada",
)
eixo_acumulado.set_ylabel("Variância acumulada (%)", fontsize=12, color="firebrick")
eixo_acumulado.tick_params(axis="y", labelcolor="firebrick")
eixo_acumulado.set_ylim(0, 105)
eixo_acumulado.axhline(80, color="gray", linestyle="--", linewidth=1, alpha=0.7)
eixo_acumulado.text(
    numero_componentes[-1], 82, "80% acumulado", fontsize=9, color="gray", ha="right"
)

for x, acumulado in zip(numero_componentes, var_acumulada_perc):
    eixo_acumulado.annotate(
        f"{acumulado:.1f}%",
        xy=(x, acumulado),
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        color="firebrick",
    )

linhas_1, rotulos_1 = eixo_individual.get_legend_handles_labels()
linhas_2, rotulos_2 = eixo_acumulado.get_legend_handles_labels()
eixo_individual.legend(linhas_1 + linhas_2, rotulos_1 + rotulos_2, loc="center right")

plt.title("Variância explicada por componente principal")
plt.tight_layout()
plt.show()
