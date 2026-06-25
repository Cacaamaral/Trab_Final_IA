import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. Carregamento dos dados
df = pd.read_csv('Classificacao_Dados.csv')

# 2. Pré-processamento
X_bruto = df.drop(columns=['id', 'categoria'])
y = df['categoria']

# Transformação de variáveis categóricas
X = pd.get_dummies(X_bruto, columns=['material'])

# 3. Divisão do conjunto de dados
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Padronização em escala
scaler = StandardScaler()
X_treino_esc = scaler.fit_transform(X_treino)
X_teste_esc = scaler.transform(X_teste)

# 5. Definição dos Modelos
# A configuração de parâmetros como max_depth e C evita o sobreajuste perfeito
modelos = {
    "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=2, random_state=42),
    "Regressao_Logistica": LogisticRegression(C=0.05, max_iter=1000, random_state=42),
    "SVM": SVC(kernel='rbf', C=0.5, probability=True, random_state=42)
}

# 6. Treinamento e Geração de Gráficos Individuais
# Utiliza um estilo limpo e profissional para os gráficos
plt.style.use('seaborn-v0_8-whitegrid')

for nome, modelo in modelos.items():
    # Inicializa uma nova figura isolada para o modelo atual
    plt.figure(figsize=(8, 6), dpi=120)
    
    # Treinamento e predição
    modelo.fit(X_treino_esc, y_treino)
    y_prob = modelo.predict_proba(X_teste_esc)
    
    # Binarização alinhada com as classes internas para cálculo da curva
    y_teste_bin = label_binarize(y_teste, classes=modelo.classes_)
    
    # Cálculo numérico do ROC via média global (micro-average)
    fpr, tpr, _ = roc_curve(y_teste_bin.ravel(), y_prob.ravel())
    roc_auc = auc(fpr, tpr)
    
    # Plotagem da curva do modelo
    plt.plot(fpr, tpr, lw=2.5, color='darkorange', label=f'Curva ROC (AUC = {roc_auc:.3f})')
    
    # Plotagem da linha de base (chute aleatório)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório (AUC = 0.500)')
    
    # Configurações de eixos e títulos
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    
    # Formata o título removendo os underlines do nome da chave
    titulo_limpo = nome.replace("_", " ")
    plt.title(f'Curva ROC Multiclasse - {titulo_limpo}', fontsize=14, fontweight='bold')
    
    plt.legend(loc="lower right", frameon=True)
    
    # Salvamento da figura com nome dinâmico
    nome_arquivo = f'curva_roc_{nome.lower()}.png'
    plt.savefig(nome_arquivo, bbox_inches='tight')
    
    # Limpa a memória e fecha a figura atual para não misturar com a próxima iteração
    plt.close()
    
    print(f"Gráfico gerado com sucesso: {nome_arquivo}")