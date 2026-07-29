"""Compara metricas para escolher o numero de clusters (K) do K-means:
metodo do cotovelo (inertia) e indice Davies-Bouldin. Pressupoe que
'dados_para_cluster' ja existe no namespace (gerado por
clusterizar_pcs.py)."""

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

faixa_k = range(3, 9)

distorcoes = []
db_kmeans = []

for k in faixa_k:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    rotulos_km = km.fit_predict(dados_para_cluster)
    # soma dos quadrados das distancias intra-grupo (WCSS) -- nao e
    # "variancia" no sentido estatistico estrito (nao normalizada por N)
    distorcoes.append(km.inertia_)
    db_kmeans.append(davies_bouldin_score(dados_para_cluster, rotulos_km))

# Davies-Bouldin: MENOR valor = clusters mais compactos e mais separados
# entre si (diferente da maioria das metricas, aqui "maior" nao e melhor)
melhor_k_db = list(faixa_k)[db_kmeans.index(min(db_kmeans))]
print(f"Melhor K pelo índice Davies-Bouldin (menor valor): K={melhor_k_db}")

# --- Metodo do cotovelo ---
plt.figure(figsize=(8, 5))
plt.plot(list(faixa_k), distorcoes, marker="o", color="navy")
plt.xlabel("Número de agrupamentos (K)")
plt.ylabel("Soma dos quadrados intra-grupo (inércia)")
plt.title("Método do Cotovelo - K-médias")
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# --- Indice Davies-Bouldin ---
cores = ["firebrick" if k == melhor_k_db else "navy" for k in faixa_k]

fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(list(faixa_k), db_kmeans, color=cores)
ax.set_title("Índice Davies-Bouldin - K-médias (menor é melhor)")
ax.set_xlabel("Número de agrupamentos (K)")
ax.set_ylabel("Índice Davies-Bouldin")
ax.annotate(
    f"menor: K={melhor_k_db}",
    xy=(melhor_k_db, db_kmeans[list(faixa_k).index(melhor_k_db)]),
    xytext=(0, 8),
    textcoords="offset points",
    ha="center",
    fontsize=10,
    color="firebrick",
)

plt.tight_layout()
plt.show()
