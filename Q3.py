import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

# 1. Carregamento dos dados
# Assume-se que as colunas se chamam 'avaliacao' (o texto) e 'sentimento' (o alvo).
# Caso o arquivo utilize nomenclaturas diferentes, altere os nomes abaixo.
df = pd.read_csv('Analise de sentimento.csv')

X = df['avaliacao']
y = df['sentimento']

# 2. Divisão do conjunto de dados (70% treinamento, 30% teste)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Vetorização do texto via TF-IDF
vetorizador = TfidfVectorizer()
X_treino_vetorizado = vetorizador.fit_transform(X_treino)
X_teste_vetorizado = vetorizador.transform(X_teste)

# 4. Definição dos Modelos de Classificação
# MultinomialNB é o padrão da indústria para classificação de textos básicos
# SVC com kernel linear é altamente eficiente em espaços de alta dimensionalidade como o TF-IDF
modelos = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', random_state=42)
}

# 5. Configuração da figura para exibição das Matrizes de Confusão lado a lado
fig, eixos = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
plt.style.use('seaborn-v0_8-whitegrid')

print("--- RESULTADOS DA ANÁLISE DE SENTIMENTO ---\n")

# 6. Treinamento e Extração de Métricas
for indice, (nome, modelo) in enumerate(modelos.items()):
    # Treinamento do modelo
    modelo.fit(X_treino_vetorizado, y_treino)
    
    # Previsão sobre o conjunto de teste
    y_previsto = modelo.predict(X_teste_vetorizado)
    
    # Cálculo das métricas quantitativas (utiliza average='weighted' para suportar múltiplas classes)
    acuracia = accuracy_score(y_teste, y_previsto)
    f1 = f1_score(y_teste, y_previsto, average='weighted')
    
    print(f"Modelo: {nome}")
    print(f"Acurácia: {acuracia:.4f}")
    print(f"F1-Score: {f1:.4f}\n")
    
    # Geração da Matriz de Confusão no eixo correspondente
    ConfusionMatrixDisplay.from_predictions(
        y_teste, 
        y_previsto, 
        ax=eixos[indice], 
        cmap='Blues',
        colorbar=False
    )
    eixos[indice].set_title(f'Matriz de Confusão - {nome}', fontsize=12, fontweight='bold')
    eixos[indice].grid(False) # Remove linhas de grade de cima da matriz para estética limpa

# 7. Salvamento das representações gráficas
plt.tight_layout()
plt.savefig('matrizes_confusao.png', bbox_inches='tight')
print("Execução finalizada. Gráfico salvo como 'matrizes_confusao.png'.")