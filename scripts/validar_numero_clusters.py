"""Plota metricas para escolher o numero de clusters (K) do K-means:
metodo do cotovelo (inercia) e indice Davies-Bouldin. Pressupoe que
'df_metricas_k' ja existe no namespace (gerado por clusterizar_pcs.py) --
os K-means de cada K ja foram treinados la, aqui so visualizamos."""

import matplotlib.pyplot as plt

faixa_k = df_metricas_k.index.tolist()
distorcoes = df_metricas_k["inercia"].tolist()
db_kmeans = df_metricas_k["davies_bouldin"].tolist()

# Davies-Bouldin: MENOR valor = clusters mais compactos e mais separados
# entre si (diferente da maioria das metricas, aqui "maior" nao e melhor)
melhor_k_db = int(df_metricas_k["davies_bouldin"].idxmin())
print(f"Melhor K pelo índice Davies-Bouldin (menor valor): K={melhor_k_db}")

# --- Metodo do cotovelo ---
plt.figure(figsize=(8, 5))
plt.plot(faixa_k, distorcoes, marker="o", color="navy")
plt.xlabel("Número de agrupamentos (K)")
plt.ylabel("Soma dos quadrados intra-grupo (inércia)")
plt.title("Método do Cotovelo - K-médias")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# --- Indice Davies-Bouldin ---
cores = ["firebrick" if k == melhor_k_db else "navy" for k in faixa_k]

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(faixa_k, db_kmeans, color=cores)
ax.set_title("Índice Davies-Bouldin - K-médias (menor é melhor)")
ax.set_xlabel("Número de agrupamentos (K)")
ax.set_ylabel("Índice Davies-Bouldin")
ax.annotate(
    f"menor: K={melhor_k_db}",
    xy=(melhor_k_db, db_kmeans[faixa_k.index(melhor_k_db)]),
    xytext=(0, 8),
    textcoords="offset points",
    ha="center",
    fontsize=10,
    color="firebrick",
)

plt.tight_layout()
plt.show()
