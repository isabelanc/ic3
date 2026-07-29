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
8. **`clusterizar_pcs.py`** — K-means no espaço dos PCs. Produz
   `dados_para_cluster`, `centroides`, e grava a coluna
   `df_com_pca["grupo_kmeans"]`.
9. **`validar_numero_clusters.py`** — método do cotovelo (inércia) e índice
   Davies-Bouldin para validar a escolha de K. Precisa de
   `dados_para_cluster`, ou seja, só roda depois do passo 8 (pelo menos
   uma vez). Se sugerir um K diferente do usado no passo 8, ajuste
   `n_clusters` em `clusterizar_pcs.py` e rode os passos 8–9 de novo.
10. **`plotar_facies_inline.py`** — mapa de fácies (grupos K-means) de um
    inline. Usa `df_com_pca` (coluna `grupo_kmeans`) e `centroides`.
11. **`plotar_facies_crossline.py`** — mesma ideia do passo 10, para uma
    crossline. Independente do passo 10.
12. **`plotar_facies_zslice.py`** — mapa de fácies em planta (inline ×
    crossline) numa fatia de tempo/profundidade fixa. Independente dos
    passos 10 e 11.

Os passos 10, 11 e 12 são autocontidos entre si (cada um define sua própria
paleta de cores fixa por grupo) — dependem apenas do passo 8 ter rodado.
