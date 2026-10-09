# Wine Quality — Pipeline de Treino com MLflow

Trabalho de IA: pipeline de treino, validação e teste com MLflow usando o dataset Wine Quality.

## 📌 Objetivo

Construir um pipeline reproduzível que:
- Carrega o dataset Wine Quality (tinto);
- Divide os dados em treino, validação e teste;
- Treina dois modelos distintos;
- Registra parâmetros, métricas e modelos no MLflow.

## 🧪 Experimentos

| Experimento | Modelo | Pergunta |
| :--- | :--- | :--- |
| 1 | Ridge Regression | Um modelo linear simples consegue prever bem a qualidade? |
| 2 | RandomForest | Um modelo não-linear captura melhor as relações? |

## 📁 Estrutura do Projeto

```text
wine-quality-mlflow/
├── data/
├── src/
│   ├── data_loader.py
│   └── train.py
├── run_experiments.py
├── requirements.txt
└── README.md

📊 Resultados
A tabela abaixo mostra o desempenho dos modelos no conjunto de teste (dados que os modelos nunca viram):

Modelo	RMSE (teste)	MAE (teste)	R² (teste)
Ridge	0.6219	0.5040	0.4082
RandomForest	0.5815	0.4553	0.4825
Conclusão: O RandomForest apresentou melhor desempenho em todas as métricas, sugerindo que as relações entre as propriedades químicas e a qualidade do vinho são predominantemente não-lineares.

🚀 Como rodar
Instale as dependências:

bash
pip install -r requirements.txt
Execute os experimentos:

bash
python run_experiments.py
Abra a UI do MLflow:

bash
mlflow ui
Acesse http://localhost:5000

🤖 Uso de IA generativa
Este projeto utilizou IA generativa (DeepSeek) para auxiliar na estruturação do pipeline, refatoração do código e configuração do MLflow.