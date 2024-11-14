
# Análise de Perfis do Instagram Usando Regressão Linear

## Descrição do Projeto

O projeto **Análise de Perfis do Instagram Usando Regressão Linear** aplica técnicas de regressão linear para prever variáveis relacionadas aos perfis do Instagram, com base em dados coletados. Utiliza o algoritmo **Ridge Regression** (regressão linear regularizada) para ajustar os dados e prever os valores de interesse. O código também implementa validação cruzada e visualizações gráficas, incluindo a linha de identidade e gráficos de resíduos para análise do desempenho do modelo.

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/arthurmarttins/analise_instagram_regressaolinear.git
   cd analise_instagram_regressaolinear
   ```

2. Crie um ambiente virtual:

   ```bash
   python -m venv venv
   ```

3. Ative o ambiente virtual:

   - No Windows:

     ```bash
     venv\Scripts\activate
     ```

   - No macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. Instale as dependências do projeto:

   ```bash
   pip install -r requirements.txt
   ```

## Como Executar

Para rodar o projeto, execute o script principal:

```bash
python main.py
```

O código irá carregar os dados, treinar o modelo de regressão, exibir gráficos de validação e imprimir a pontuação média da validação cruzada.

### Exemplo de Execução

1. O modelo carregará automaticamente os dados de entrada e começará a execução.
2. Um gráfico será gerado mostrando os resultados da regressão.
3. A pontuação média da validação cruzada será impressa no terminal.

## Estrutura dos Arquivos

- `/main.py`: Código principal que realiza o carregamento dos dados, treinamento do modelo e avaliação.
- `/requirements.txt`: Lista de dependências necessárias para o projeto.
- `/data/`: Pasta onde os dados de entrada devem ser colocados (caso seja necessário).
- `/results/`: Pasta onde gráficos e resultados de execução podem ser salvos (se implementado).
- `/docs/`: Relatório

## Tecnologias Utilizadas

- **Python**: Linguagem de programação principal.
- **Scikit-Learn**: Para o modelo de regressão e validação cruzada.
- **Matplotlib**: Para a visualização de gráficos.
- **NumPy**: Para manipulação de arrays e dados numéricos.
- **Pandas**: Para manipulação de dados.

## Autores e Colaboradores

- **Arthur Lago Martins**
- **João Victor Oliveira Santos**