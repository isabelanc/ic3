"""Agrupa (K-means) as amostras no espaço dos componentes principais, para
cada K de uma faixa de valores testados. Pressupoe que 'df_pca' e
'df_com_pca' ja existem no namespace (gerados por aplicar_pca.py).

Produz:
- df_metricas_k: DataFrame indexado por K, com a inercia e o indice
  Davies-Bouldin de cada K testado (usado por validar_numero_clusters.py).
- df_grupos_por_k: DataFrame com o mesmo indice de df_pca, uma coluna
  "grupo_k{K}" por valor de K testado, com os rotulos de cluster (NaN
  preservado nas linhas de borda que o dropna() excluiu do treino).
- centroides_por_k: dict {K: cluster_centers_} de cada K testado.

Essas tres estruturas guardam o resultado de TODOS os K testados, para
poder trocar de K nos plots de facies sem precisar re-treinar. Para usar
um K especifico (o codigo abaixo ja faz isso para k_escolhido):

    k_escolhido = 4
    df_com_pca["grupo_kmeans"] = df_grupos_por_k[f"grupo_k{k_escolhido}"]
    centroides = centroides_por_k[k_escolhido]
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

colunas_pca = ["PC1", "PC2", "PC3", "PC4", "PC5"]
faixa_k = range(3, 9)

dados_para_cluster = df_pca[colunas_pca].dropna()

metricas_k = {}
centroides_por_k = {}
colunas_grupo_por_k = {}

for k in faixa_k:
    # n_init=50: mesma robustez usada antes so no K final, agora aplicada a
    # todos os K testados -- os rotulos de cada K aqui ja sao os mesmos que
    # serao reaproveitados na plotagem, nao servem so de rascunho para o
    # metodo do cotovelo
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    labels = kmeans.fit_predict(dados_para_cluster)

    metricas_k[k] = {
        # soma dos quadrados das distancias intra-grupo (WCSS) -- nao e
        # "variancia" no sentido estatistico estrito (nao normalizada por N)
        "inercia": kmeans.inertia_,
        "davies_bouldin": davies_bouldin_score(dados_para_cluster, labels),
    }

    # rotulos com o mesmo indice de dados_para_cluster (que perdeu linhas no
    # dropna), reindexados de volta ao indice completo de df_pca, para
    # preservar NaN nas linhas de borda excluidas
    rotulos = pd.Series(labels, index=dados_para_cluster.index)
    colunas_grupo_por_k[f"grupo_k{k}"] = rotulos.reindex(df_pca.index)

    centroides_por_k[k] = kmeans.cluster_centers_

df_metricas_k = pd.DataFrame.from_dict(metricas_k, orient="index")
df_metricas_k.index.name = "k"

df_grupos_por_k = pd.DataFrame(colunas_grupo_por_k, index=df_pca.index)

print(df_metricas_k)

# Escolha do K final para uso nos plots de facies (plotar_facies_inline.py /
# plotar_facies_crossline.py / plotar_facies_zslice.py). Escolhido
# manualmente como 4; para referencia, o K de menor Davies-Bouldin tambem
# e impresso abaixo.
melhor_k_db = int(df_metricas_k["davies_bouldin"].idxmin())
print(f"K de menor Davies-Bouldin (referência): {melhor_k_db}")

k_escolhido = 4
print(f"K escolhido para os plots de fácies: {k_escolhido}")

df_com_pca["grupo_kmeans"] = df_grupos_por_k[f"grupo_k{k_escolhido}"]
centroides = centroides_por_k[k_escolhido]
