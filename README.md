# Tech Challenge — Fase 3 · Predição e Inteligência Analítica para Alfabetização no Brasil

> Modelo supervisionado que prevê se uma criança do 2º ano do fundamental será
> **alfabetizada ou não**, a partir de variáveis educacionais, territoriais e
> socioeconômicas da camada **Gold** construída na Fase 2.

Pós Tech em Machine Learning Engineering — FIAP · Tech Challenge Fase 3.

---

## 1. Contexto do problema

A alfabetização na idade certa é um dos principais indicadores do desenvolvimento
educacional do país. Olhar apenas o indicador atual não basta para decidir: gestores
públicos precisam **antecipar risco**, identificar regiões vulneráveis e entender quais
fatores mais pesam no resultado. Este projeto transforma os dados públicos integrados na
Fase 2 em inteligência analítica aplicada à tomada de decisão.

## 2. Objetivo analítico

_(a preencher)_ — definição formal do alvo, unidade de análise e recorte temporal.

## 3. Descrição da base utilizada

Fonte primária: camada **Gold** da Fase 2 (`data/gold/`), em especial a feature store
`features_ml_municipio` (16.713 registros, 27 colunas), que já é construída **sem
vazamento** — nenhuma variável derivada do alvo entra como atributo.

Enriquecimento externo possível (previsto no enunciado): IBGE, Censo Escolar, FUNDEB,
Atlas do Desenvolvimento Humano, PNAD, Cadastro Único.

> A procedência de cada tabela e a data do snapshot estão em [`data/README.md`](data/README.md).

## 4. Etapas de modelagem

_(a preencher)_ — pipeline Scikit-learn: imputação, transformação de variáveis numéricas e
categóricas, tratamento de data leakage, pré-processamento integrado ao modelo, treino,
validação e garantia de replicabilidade/generalização.

## 5. Escolha do algoritmo

_(a preencher)_

## 6. Métricas de avaliação

_(a preencher)_

## 7. Interpretação dos resultados

_(a preencher)_ — Feature Importance e SHAP Values.

## 8. Insights encontrados

_(a preencher)_ — responde às perguntas de negócio: quais fatores mais impactam a
alfabetização, quais municípios têm maior risco, quais regiões têm padrões semelhantes,
como prever quem pode não atingir metas futuras.

## 9. Limitações do projeto

_(a preencher)_

## 10. Aplicação prática para políticas públicas

_(a preencher)_

## 11. Possíveis evoluções futuras

_(a preencher)_

---

## Como reproduzir

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Pipeline completa: pré-processamento -> treino -> avaliação
python -m src.modeling.treinar
```

## Estrutura do repositório

```
tech-challenge-fase3/
├── data/            # snapshots da camada Gold (origem documentada em data/README.md)
├── notebooks/       # 01 EDA · 02 pré-processamento · 03 modelagem · 04 interpretação
├── src/
│   ├── preprocessing/   # imputação, encoding, feature engineering (ColumnTransformer)
│   ├── modeling/        # definição, treino e tuning dos modelos
│   ├── evaluation/      # métricas, validação, curvas
│   └── visualization/   # gráficos de EDA e de resultado
├── reports/         # relatório analítico e decisões (ADRs)
├── images/          # figuras geradas para o README e o relatório
├── requirements.txt
├── README.md
└── .gitignore
```
