import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# 1. Carregando os dados
df = pd.read_csv('Books.csv', encoding='utf-8', sep=None, engine='python')

# 2. Configurações de colunas (ajuste os nomes se necessário)
coluna_titulo = 'Book-Title'
coluna_texto = 'Book-Author' # Use 'Summary' ou 'Description' se tiver no arquivo!

# Pegando uma amostra maior para o gráfico ficar bonito
df = df[[coluna_titulo, coluna_texto]].head(200).dropna()

# 3. Álgebra Linear (Vetorização e Redução)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df[coluna_texto])

# SVD para reduzir a 2 dimensões
svd = TruncatedSVD(n_components=2)
pontos_2d = svd.fit_transform(tfidf_matrix)

# 4. Criando um novo DataFrame para o gráfico
df_plot = pd.DataFrame({
    'Título': df[coluna_titulo],
    'Autor': df[coluna_texto],
    'x': pontos_2d[:, 0],
    'y': pontos_2d[:, 1]
})

# 5. Visualização Interativa com Plotly
fig = px.scatter(
    df_plot, x='x', y='y', 
    hover_name='Título', 
    hover_data=['Autor'],
    title='Exploração Semântica de Livros (SVD + NLP)',
    color_discrete_sequence=['#636EFA']
)

# Estilizando para ficar mais "Dark Academia"
fig.update_layout(
    template="plotly_dark",
    xaxis_title="Proximidade por Tópico A",
    yaxis_title="Proximidade por Tópico B"
)

fig.show()