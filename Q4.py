import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Carregamento dos dados
df = pd.read_csv('Apartamentos_meier.csv')

# 2. Separacao entre variaveis independentes (X) e dependente (y)
X = df.drop(columns=['preco'])
y = df['preco']

# 3. Divisao do conjunto de dados (70% treino, 30% teste)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Inicializacao e Treinamento do Modelo
modelo_regressao = LinearRegression()
modelo_regressao.fit(X_treino, y_treino)

# 5. Execucao das previsoes com a base de teste
y_previsto = modelo_regressao.predict(X_teste)

# 6. Avaliacao do Modelo
mae = mean_absolute_error(y_teste, y_previsto)
rmse = np.sqrt(mean_squared_error(y_teste, y_previsto))
r2 = r2_score(y_teste, y_previsto)

print("--- METRICAS DE AVALIACAO ---")
print(f"MAE (Erro Absoluto Medio): {mae:.2f}")
print(f"RMSE (Raiz do Erro Quadratico Medio): {rmse:.2f}")
print(f"R2-Score: {r2:.4f}")

# 7. Geracao de grafico comparativo (Real vs Previsto)
plt.figure(figsize=(8, 6), dpi=120)
plt.style.use('seaborn-v0_8-whitegrid')

# Plota os pontos previstos em relacao aos dados reais
plt.scatter(y_teste, y_previsto, color='royalblue', alpha=0.7, edgecolors='k')

# Traca a linha de previsao perfeita (y = x)
plt.plot([y_teste.min(), y_teste.max()], [y_teste.min(), y_teste.max()], 'r--', lw=2, label='Previsao Perfeita')

plt.xlabel('Valor Real', fontsize=12, fontweight='bold')
plt.ylabel('Valor Previsto', fontsize=12, fontweight='bold')
plt.title('Regressao Linear - Valor Real vs Previsto', fontsize=14, fontweight='bold')
plt.legend(frameon=True)

# Salva o resultado em formato PNG
plt.savefig('regressao_linear.png', bbox_inches='tight')
print("Grafico salvo como 'regressao_linear.png'.")