import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import os


def convert_to_numeric(value):
    if isinstance(value, str):
        value = value.replace(',', '')  
        value = value.strip().lower()  

        if 'k' in value:
            try:
                return float(value.replace('k', '')) * 1000  #
            except ValueError:
                return None
        elif 'm' in value:
            try:
                return float(value.replace('m', '')) * 1000000
            except ValueError:
                return None
        elif 'b' in value:
            try:
                return float(value.replace('b', '')) * 1000000000
            except ValueError:
                return None
        else:
            try:
                return float(value)  
            except ValueError:
                return None 
    return value  

def convert_percentage_to_numeric(value):
    if isinstance(value, str):
        value = value.strip().lower()  
        if '%' in value:
            try:
                return float(value.replace('%', '')) / 100  
            except ValueError:
                return None
        else:
            try:
                return float(value)  
            except ValueError:
                return None 
    return value 


# Carregar dados
df = pd.read_csv('data/top_insta_influencers_data.csv')

# Exibir as primeiras linhas dos dados
print(df.head())

# Conversões
df['followers'] = pd.to_numeric(df['followers'].apply(convert_to_numeric), errors='coerce')
df['avg_likes'] = pd.to_numeric(df['avg_likes'].apply(convert_to_numeric), errors='coerce')
df['posts'] = pd.to_numeric(df['posts'].apply(convert_to_numeric), errors='coerce')
df['new_post_avg_like'] = pd.to_numeric(df['new_post_avg_like'].apply(convert_to_numeric), errors='coerce')
df['total_likes'] = pd.to_numeric(df['total_likes'].apply(convert_to_numeric), errors='coerce')
df['60_day_eng_rate'] = pd.to_numeric(df['60_day_eng_rate'].apply(convert_percentage_to_numeric), errors='coerce')
df['60_day_eng_rate'] = df['60_day_eng_rate'].fillna(0)

# Selecionar apenas colunas numéricas
df_numeric = df.select_dtypes(include=['float64', 'int64'])

# Verifique se a pasta "results" existe; caso contrário, crie-a
if not os.path.exists('results'):
    os.makedirs('results')

# 1. Mapa de calor de correlação entre variáveis
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Mapa de Calor das Correlações entre Variáveis")
plt.savefig('results/heatmap_correlations.png') 
plt.close()  

# 2. Histograma - Distribuição de seguidores
plt.figure(figsize=(8, 6))
sns.histplot(df['followers'], bins=30, kde=True)
plt.title("Distribuição de Seguidores")
plt.xlabel("Seguidores")
plt.ylabel("Frequência")
plt.savefig('results/histogram_followers.png')  
plt.close()  

# 3. Gráfico de Dispersão - Seguidores vs. Média de Likes
plt.figure(figsize=(8, 6))
sns.scatterplot(x='followers', y='avg_likes', data=df)
plt.title("Seguidores vs. Média de Likes")
plt.xlabel("Seguidores")
plt.ylabel("Média de Likes")
plt.savefig('results/scatter_followers_vs_likes.png')  
plt.close() 

# 4. Boxplot - Score de Influência por País
plt.figure(figsize=(10, 6))
sns.boxplot(x='country', y='influence_score', data=df)
plt.xticks(rotation=90)
plt.title("Score de Influência por País")
plt.xlabel("País")
plt.ylabel("Score de Influência")
plt.savefig('results/boxplot_influence_score_by_country.png')  
plt.close() 

# Definir as variáveis independentes
X = df[['followers', 'rank', 'influence_score', 'posts', 'country']]

# Converter a variável categórica 'country' em variáveis dummies
X = pd.get_dummies(X, columns=['country'], drop_first=True)

# Definir a variável dependente
y = df['60_day_eng_rate']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializar o modelo 
model = Ridge(alpha=1.0)  

# Treinar o modelo
model.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Impressão
print(f'R²: {r2}')
print(f'MSE: {mse}')
print(f'MAE: {mae}')

# Interpretação dos Coeficientes
print(f'Coeficientes: {model.coef_}')
print(f'Intercepto: {model.intercept_}')


# Validação Cruzada
scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print(f'Pontuação média de validação cruzada: {scores.mean()}')
