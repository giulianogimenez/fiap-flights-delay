# Flight Delays Tech Challenge

Projeto de Machine Learning Engineering para analisar e prever atrasos de voos nos EUA usando a base `flights.csv`.

## Objetivo

O trabalho responde três frentes do desafio:

- EDA: estatísticas descritivas, valores ausentes, visualizações e padrões por companhia, aeroporto e horário.
- Aprendizado supervisionado: classificação de atraso (`ARRIVAL_DELAY > 15`) como tarefa principal e regressão de minutos de atraso como extensão.
- Aprendizado não supervisionado: clusterização de aeroportos de origem por perfil de volume, atraso, cancelamento, diversidade de rotas e companhias.

## Estrutura

```text
.
├── data/                  # instruções para posicionar dados locais
├── docs/                  # dicionário e decisões de modelagem
├── notebooks/             # notebooks da análise
├── reports/               # tabelas, figuras e resumo executivo gerados
├── scripts/               # execução ponta a ponta
├── src/flight_delays/     # código reutilizável do projeto
└── tests/                 # testes unitários
```

## Como Rodar

Crie o ambiente e instale dependências:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Coloque os arquivos em `data/raw/` ou deixe-os em `~/Downloads` com os nomes:

- `flights.csv`
- `airlines.csv`
- `airports.csv`

Execute a pipeline com uma amostra:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --nrows 50000
```

Para usar o dataset completo:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_pipeline.py --nrows 0
```

## Resultados Gerados

A última execução local usou `50.000` linhas de `flights.csv`.

- Melhor classificador: `Logistic Regression`, F1 `0.574`, ROC-AUC `0.751`.
- Segundo classificador: `Random Forest`, F1 `0.570`, ROC-AUC `0.750`.
- Melhor regressor: `Random Forest Regressor`, MAE `21.86` minutos, RMSE `40.47` minutos.
- Segundo regressor: `Ridge Regression`, MAE `22.98` minutos, RMSE `41.19` minutos.

Consulte detalhes em:

- `reports/executive_summary.md`
- `reports/tables/classification_metrics.csv`
- `reports/tables/regression_metrics.csv`
- `reports/tables/cluster_summary.csv`
- `reports/figures/`

## Notebooks

- `notebooks/01_eda.ipynb`: exploração dos dados, ausentes, estatísticas e gráficos.
- `notebooks/02_supervised_classification.ipynb`: classificação de atraso com dois modelos.
- `notebooks/03_supervised_regression.ipynb`: regressão do tempo de atraso com dois modelos.
- `notebooks/04_unsupervised.ipynb`: clusterização de perfis de aeroportos.

## Decisão Contra Vazamento de Dados

Os modelos supervisionados usam apenas variáveis conhecidas antes do voo: data, companhia, origem, destino, rota, horários programados, distância e tempo previsto. Colunas observadas depois do voo, como `DEPARTURE_DELAY`, `ARRIVAL_TIME`, `TAXI_OUT`, `AIR_TIME` e causas de atraso, ficam fora das features.

## Limitações e Próximos Passos

- Validar os resultados no dataset completo, não apenas na amostra.
- Testar validação temporal por múltiplos meses para medir robustez fora da amostra.
- Incluir dados externos, como clima e feriados, para melhorar previsão de eventos extremos.
- Explorar modelos de boosting e calibração de probabilidades para uso operacional.
