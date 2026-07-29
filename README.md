# Pipeline de atributos sísmicos

Scripts em `scripts/` para carregar atributos sísmicos (SEGY), reduzir
dimensionalidade (PCA) e segmentar fácies sísmicas (K-means). Cada script
assume que os scripts anteriores já rodaram na mesma sessão (notebook/REPL)
e deixaram as variáveis necessárias no namespace — não são módulos
independentes, exceto pelas funções/config reaproveitadas via `import`.

## Ordem de execução

1. **`carregar_atributos_sismicos.py`** — lê os volumes SEGY e monta
   `df_final` (inline, xline, sample_idx, sample_time + um atributo por
   coluna). Marca como `NaN` as bordas de topo/base não geradas pelo
   OpendTect.
2. **`normalizar_atributos.py`** — normalização z-score por atributo,
   preservando os `NaN` de borda. Produz `df_normalizado`.
3. **`plotar_secao_inline.py`** *(opcional/exploratório)* — plota a seção de
   um inline para todos os atributos brutos. Só depende dos SEGY originais,
   pode rodar a qualquer momento após o passo 1.
4. **`aplicar_pca.py`** — PCA sobre `df_normalizado`. Produz `df_com_pca`
   (DataFrame completo + colunas `PC1..PCn`), `modelo_pca` e `df_pca` (só as
   colunas de PC, reindexadas).
5. **`plotar_scree.py`** — variância individual e acumulada por componente,
   para decidir quantos PCs faz sentido reter. Usa `modelo_pca`.
6. **`heatmap_correlacao_pcs.py`** — correlação entre os atributos originais
   e os PCs. Usa `df_com_pca`.
7. **`plotar_secao_inline_pcs.py`** — seção de um inline para cada PC. Usa
   `df_com_pca`.
8. **`clusterizar_pcs.py`** — treina um K-means para **cada K de 3 a 8** (uma
   única vez, sem duplicar o treino no passo 9). Produz:
   - `df_metricas_k` — inércia e índice Davies-Bouldin por K;
   - `df_grupos_por_k` — uma coluna `grupo_k{K}` por K testado (mesmo índice
     de `df_pca`, `NaN` preservado nas linhas de borda);
   - `centroides_por_k` — dict `{K: cluster_centers_}`.

   Ao final, escolhe `k_escolhido` (por padrão, o K de menor Davies-Bouldin)
   e grava `df_com_pca["grupo_kmeans"]` / `centroides` a partir dele, para uso
   direto nos plots de fácies. Para trocar de K sem re-treinar, é só repetir
   as 3 últimas linhas do script com outro valor de `k_escolhido`:
   ```python
   k_escolhido = 4
   df_com_pca["grupo_kmeans"] = df_grupos_por_k[f"grupo_k{k_escolhido}"]
   centroides = centroides_por_k[k_escolhido]
   ```
9. **`validar_numero_clusters.py`** — plota o método do cotovelo (inércia) e
   o índice Davies-Bouldin a partir de `df_metricas_k` (não re-treina nada,
   só visualiza o que o passo 8 já calculou para todos os K).
10. **`plotar_facies_inline.py`** — mapa de fácies (grupos K-means) de um
    inline, para o `k_escolhido`. Usa `df_com_pca` (coluna `grupo_kmeans`) e
    `centroides`.
11. **`plotar_facies_crossline.py`** — mesma ideia do passo 10, para uma
    crossline. Independente do passo 10.
12. **`plotar_facies_zslice.py`** — mapa de fácies em planta (inline ×
    crossline) numa fatia de tempo/profundidade fixa. Independente dos
    passos 10 e 11.
13. **`plotar_facies_inline_todos_k.py`** — mesma ideia do passo 10, mas um
    subplot para **cada K testado** (3 a 8), lado a lado, pra comparar
    visualmente como a segmentação muda com o número de grupos. Usa
    `df_grupos_por_k`/`centroides_por_k` (passo 8) diretamente — não precisa
    do `k_escolhido`, nem de re-treinar.

Os passos 10, 11 e 12 são autocontidos entre si (cada um define sua própria
paleta de cores fixa por grupo) — dependem apenas do passo 8 ter rodado. O
passo 13 usa uma paleta com 8 cores (para cobrir até K=8); se `faixa_k` em
`clusterizar_pcs.py` mudar para K maior, a paleta precisa crescer junto.

## Alternativa de clusterização: Gaussian Mixture Model (GMM)

Mesma ideia dos passos 8–9, usando GMM em vez de K-means (não substitui,
roda em paralelo — ambos podem coexistir na mesma sessão, usam nomes de
variável diferentes). Precisa de `df_pca`/`df_com_pca` (passo 4).

- **`clusterizar_gmm.py`** — treina um `GaussianMixture` (`covariance_type=
  "full"`, `n_init=5`) para cada K de 3 a 8, com checagem de convergência do
  EM por K. Produz `df_metricas_gmm_k` (BIC, Davies-Bouldin, convergência),
  `df_grupos_gmm_por_k` (rótulos por K, `NaN` de borda preservado),
  `responsabilidades_gmm_por_k` (dict `{K: DataFrame}` com a probabilidade
  de cada amostra pertencer a cada componente) e `modelos_gmm_por_k` (dict
  `{K: GaussianMixture}`). Ao final, escolhe `k_escolhido_gmm` (por padrão,
  o K de menor BIC — critério padrão para número de componentes de misturas
  gaussianas) e grava `df_com_pca["grupo_gmm"]`, além de `pesos_pi`,
  `medias_mu`, `covariancias` do modelo escolhido.
- **`validar_numero_clusters_gmm.py`** — plota BIC e Davies-Bouldin a partir
  de `df_metricas_gmm_k`, com o melhor K destacado em cada gráfico (sem
  re-treinar). Avisa se algum K não convergiu.

Os scripts de plot de fácies (10–12) ainda só leem a coluna `grupo_kmeans` —
para plotar fácies do GMM, adapte-os para usar `grupo_gmm` (mesma lógica de
`ListedColormap`/`BoundaryNorm`, trocando a coluna e `centroides` por
`medias_mu`).
