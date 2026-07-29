"""Agrupa (K-means) as amostras no espaço dos componentes principais.
Pressupoe que 'df_pca' e 'df_com_pca' ja existem no namespace (gerados por
aplicar_pca.py)."""

import pandas as pd
from sklearn.cluster import KMeans

colunas_pca = ["PC1", "PC2", "PC3", "PC4", "PC5"]

dados_para_cluster = df_pca[colunas_pca].dropna()

# n_clusters = K
# random_state = 42 (garante que o agrupamento aleatório inicial seja sempre o mesmo)
# n_init = 50 (roda o algoritmo 50 vezes com inícios diferentes e escolhe o melhor)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=50)

# .fit() executa todo loop de cálculo de distâncias, reatribuição de rótulos e recálculo de centroides
kmeans.fit(dados_para_cluster)

# rotulos com o mesmo indice de dados_para_cluster (que perdeu linhas no
# dropna), para poder reindexar com seguranca de volta ao DataFrame
# completo, preservando NaN nas linhas de borda excluidas
rotulos = pd.Series(kmeans.labels_, index=dados_para_cluster.index, name="cluster")
rotulos_completo = rotulos.reindex(df_pca.index)

# grava o rotulo de cluster no DataFrame completo, para uso nos plots de
# facies (plotar_facies_inline.py / plotar_facies_crossline.py)
df_com_pca["grupo_kmeans"] = rotulos_completo

# posições finais dos 4 centros (cada um com 5 coordenadas), depois de todas as repetições (mu_k)
centroides = kmeans.cluster_centers_

# soma dos quadrados das distancias de cada amostra ao centroide do seu
# cluster (inertia_) -- NAO e "variancia" no sentido estatistico estrito:
# nao e normalizada pelo numero de amostras, entao cresce com N e cai
# (ou mantem) conforme K aumenta; nao e diretamente comparavel entre
# datasets/subconjuntos de tamanhos diferentes sem normalizar
inercia_final = kmeans.inertia_

print(f"Soma dos quadrados intra-grupo (inércia): {inercia_final:.4f}")
