"""Agrupa (Gaussian Mixture Model) as amostras no espaço dos componentes
principais, para cada K de uma faixa de valores testados. Pressupoe que
'df_pca' e 'df_com_pca' ja existem no namespace (gerados por
aplicar_pca.py).

Produz:
- df_metricas_gmm_k: DataFrame indexado por K, com BIC, indice
  Davies-Bouldin e status de convergencia do EM de cada K testado (usado
  por validar_numero_clusters_gmm.py).
- df_grupos_gmm_por_k: DataFrame com o mesmo indice de df_pca, uma coluna
  "grupo_k{K}" por K testado (hard clustering / gmm.predict), NaN
  preservado nas linhas de borda que o dropna() excluiu do treino.
- responsabilidades_gmm_por_k: dict {K: DataFrame} com a probabilidade de
  cada amostra pertencer a cada componente (gmm.predict_proba), mesmo
  indice de df_pca, NaN preservado.
- modelos_gmm_por_k: dict {K: GaussianMixture ja ajustado}.

Para usar um K especifico (o codigo abaixo ja faz isso para
k_escolhido_gmm):

    k_escolhido_gmm = 4
    df_com_pca["grupo_gmm"] = df_grupos_gmm_por_k[f"grupo_k{k_escolhido_gmm}"]
    modelo = modelos_gmm_por_k[k_escolhido_gmm]
    pesos_pi, medias_mu, covariancias = modelo.weights_, modelo.means_, modelo.covariances_
"""

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score

colunas_pca = ["PC1", "PC2", "PC3", "PC4", "PC5"]
faixa_k = range(3, 9)

dados_para_cluster = df_pca[colunas_pca].dropna()

metricas_gmm_k = {}
modelos_gmm_por_k = {}
colunas_grupo_gmm_por_k = {}
responsabilidades_gmm_por_k = {}

for k in faixa_k:
    # covariance_type='full': matriz de covariancia (Sigma_k) livre por
    # componente (elipses geologicas); n_init=5, mesma robustez usada no
    # K-means, aplicada de forma consistente tanto para o BIC quanto para
    # o Davies-Bouldin (evita fits com parametros diferentes por metrica)
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=5)
    gmm.fit(dados_para_cluster)

    if not gmm.converged_:
        print(f"AVISO: EM nao convergiu para K={k} (n_iter_={gmm.n_iter_})")

    labels = gmm.predict(dados_para_cluster)
    probs = gmm.predict_proba(dados_para_cluster)

    metricas_gmm_k[k] = {
        "bic": gmm.bic(dados_para_cluster),
        "davies_bouldin": davies_bouldin_score(dados_para_cluster, labels),
        "convergiu": gmm.converged_,
        "n_iter": gmm.n_iter_,
    }

    rotulos = pd.Series(labels, index=dados_para_cluster.index)
    colunas_grupo_gmm_por_k[f"grupo_k{k}"] = rotulos.reindex(df_pca.index)

    df_probs = pd.DataFrame(
        probs, index=dados_para_cluster.index,
        columns=[f"prob_grupo_{i}" for i in range(k)],
    )
    responsabilidades_gmm_por_k[k] = df_probs.reindex(df_pca.index)

    modelos_gmm_por_k[k] = gmm

df_metricas_gmm_k = pd.DataFrame.from_dict(metricas_gmm_k, orient="index")
df_metricas_gmm_k.index.name = "k"

df_grupos_gmm_por_k = pd.DataFrame(colunas_grupo_gmm_por_k, index=df_pca.index)

print(df_metricas_gmm_k)

# Escolha do K final. BIC e o criterio padrao para numero de componentes de
# uma mistura gaussiana (e o que o proprio ajuste otimiza, penalizado pela
# complexidade do modelo); Davies-Bouldin fica como referencia cruzada.
melhor_k_bic = int(df_metricas_gmm_k["bic"].idxmin())
melhor_k_db_gmm = int(df_metricas_gmm_k["davies_bouldin"].idxmin())
print(f"K de menor BIC: {melhor_k_bic} | K de menor Davies-Bouldin: {melhor_k_db_gmm}")

k_escolhido_gmm = melhor_k_bic
print(f"K escolhido para o GMM: {k_escolhido_gmm}")

df_com_pca["grupo_gmm"] = df_grupos_gmm_por_k[f"grupo_k{k_escolhido_gmm}"]

modelo_gmm_escolhido = modelos_gmm_por_k[k_escolhido_gmm]
rotulos_gmm = df_grupos_gmm_por_k[f"grupo_k{k_escolhido_gmm}"]
responsabilidades = responsabilidades_gmm_por_k[k_escolhido_gmm]
pesos_pi = modelo_gmm_escolhido.weights_
medias_mu = modelo_gmm_escolhido.means_
covariancias = modelo_gmm_escolhido.covariances_
