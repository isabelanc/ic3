"""Plota metricas para escolher o numero de componentes (K) do GMM: BIC e
indice Davies-Bouldin. Pressupoe que 'df_metricas_gmm_k' ja existe no
namespace (gerado por clusterizar_gmm.py) -- os GMM de cada K ja foram
ajustados la, aqui so visualizamos."""

import matplotlib.pyplot as plt

faixa_k = df_metricas_gmm_k.index.tolist()
valores_bic = df_metricas_gmm_k["bic"].tolist()
db_gmm = df_metricas_gmm_k["davies_bouldin"].tolist()

nao_convergiu = df_metricas_gmm_k.index[~df_metricas_gmm_k["convergiu"]].tolist()
if nao_convergiu:
    print(f"AVISO: EM nao convergiu para K={nao_convergiu} -- BIC/Davies-Bouldin desses K podem nao ser confiaveis")

# BIC: MENOR valor = melhor tradeoff entre ajuste e complexidade do modelo
melhor_k_bic = int(df_metricas_gmm_k["bic"].idxmin())
print(f"Melhor K pelo BIC (menor valor): K={melhor_k_bic}")

# --- BIC ---
plt.figure(figsize=(8, 5))
plt.plot(faixa_k, valores_bic, marker="o", color="navy")
plt.scatter([melhor_k_bic], [valores_bic[faixa_k.index(melhor_k_bic)]], color="firebrick", zorder=3, label=f"menor BIC: K={melhor_k_bic}")
plt.xlabel("Número de agrupamentos (K)")
plt.ylabel("BIC")
plt.title("Otimização do número de agrupamentos - Critério de Informação Bayesiano (BIC)")
plt.legend()
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# --- Indice Davies-Bouldin: MENOR valor = clusters mais compactos e mais
# separados entre si (diferente da maioria das metricas, aqui "maior" nao e
# melhor) ---
melhor_k_db = int(df_metricas_gmm_k["davies_bouldin"].idxmin())
print(f"Melhor K pelo índice Davies-Bouldin (menor valor): K={melhor_k_db}")

cores = ["firebrick" if k == melhor_k_db else "navy" for k in faixa_k]

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(faixa_k, db_gmm, color=cores)
ax.set_title("Índice Davies-Bouldin - GMM (menor é melhor)")
ax.set_xlabel("Número de agrupamentos (K)")
ax.set_ylabel("Índice Davies-Bouldin")
ax.annotate(
    f"menor: K={melhor_k_db}",
    xy=(melhor_k_db, db_gmm[faixa_k.index(melhor_k_db)]),
    xytext=(0, 8),
    textcoords="offset points",
    ha="center",
    fontsize=10,
    color="firebrick",
)

plt.tight_layout()
plt.show()
