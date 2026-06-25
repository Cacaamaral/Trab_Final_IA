import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# 1. Carregamento dos dados
df = pd.read_csv('Agrupamento.csv')

# Removemos possíveis colunas de identificação (como 'id' ou 'cliente_id') 
# que não devem entrar no cálculo matemático de distância
colunas_remover = [col for col in df.columns if col.lower() in ['id', 'clienteid', 'customerid']]
X_bruto = df.drop(columns=colunas_remover)

# Mantemos apenas variáveis numéricas
X_numerico = X_bruto.select_dtypes(include=['int64', 'float64'])

# 2. Padronização (Z-score) - Exigência do professor
scaler = StandardScaler()
X_padronizado = scaler.fit_transform(X_numerico)

# 3. Agglomerative Hierarchical Clustering (Dendrograma)
plt.figure(figsize=(10, 6), dpi=120)
# O método 'ward' minimiza a variância dentro dos clusters
dendrograma = sch.dendrogram(sch.linkage(X_padronizado, method='ward'))
plt.title('Dendrograma - Agrupamento Hierárquico', fontsize=14, fontweight='bold')
plt.xlabel('Índices dos Registros')
plt.ylabel('Distância Euclidiana (Ward)')
plt.savefig('dendrograma.png', bbox_inches='tight')
plt.close()
print("Etapa 1: Dendrograma gerado e salvo como 'dendrograma.png'.")

# 4. Configuração do K-Means
# ATENÇÃO: Defini 3 clusters como padrão inicial. 
# Olhe a imagem do dendrograma: se as linhas verticais mais longas sugerirem
# 4 ou 5 grupos, altere a variável abaixo e rode o código novamente!
numero_clusters = 3 

kmeans = KMeans(n_clusters=numero_clusters, random_state=42, n_init=10)
grupos = kmeans.fit_predict(X_padronizado)
df['Grupo_KMeans'] = grupos

# 5. Redução de Dimensionalidade (PCA) e Visualização Gráfica
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_padronizado)

plt.figure(figsize=(8, 6), dpi=120)
plt.style.use('seaborn-v0_8-whitegrid')

# Plota os pontos reais (coloridos por grupo)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=grupos, cmap='viridis', s=60, edgecolors='k', alpha=0.8)

# Plota os centroides do K-Means em vermelho
centroides_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1], c='red', s=200, marker='X', label='Centroides')

plt.title(f'Grupos K-Means (k={numero_clusters}) - Projeção PCA 2D', fontsize=14, fontweight='bold')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()

plt.savefig('kmeans_grupos.png', bbox_inches='tight')
plt.close()
print(f"Etapa 2: Gráfico do K-Means (k={numero_clusters}) salvo como 'kmeans_grupos.png'.")

# Mostra a distribuição de itens por grupo no terminal para usarmos nos comentários finais
print("\n--- DISTRIBUIÇÃO DOS GRUPOS ---")
print(df['Grupo_KMeans'].value_counts().sort_index())