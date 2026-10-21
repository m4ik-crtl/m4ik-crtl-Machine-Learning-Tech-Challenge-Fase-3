"""Definição dos modelos candidatos, cada um com seu pré-processamento acoplado.

Três famílias, do mais simples/interpretável ao mais expressivo:

- Regressão Logística — baseline linear, transparente.
- Random Forest — não-linear, robusto, boa linha de base de ensemble.
- Histogram Gradient Boosting — boosting nativo do Scikit-learn, costuma dar o
  melhor desempenho tabular e lida bem com o volume.

Cada modelo é um `Pipeline` (pré-processamento + estimador) para garantir que o
tratamento seja ajustado só no treino e viaje junto do estimador.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import construir_preprocessador

SEMENTE = 42


def modelos_candidatos() -> dict[str, Pipeline]:
    return {
        "regressao_logistica": Pipeline([
            ("prep", construir_preprocessador(padronizar=True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEMENTE)),
        ]),
        "random_forest": Pipeline([
            ("prep", construir_preprocessador(padronizar=False)),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=50,
                n_jobs=-1, random_state=SEMENTE,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("prep", construir_preprocessador(padronizar=False)),
            ("clf", HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.08, max_depth=6,
                l2_regularization=1.0, random_state=SEMENTE,
            )),
        ]),
    }
