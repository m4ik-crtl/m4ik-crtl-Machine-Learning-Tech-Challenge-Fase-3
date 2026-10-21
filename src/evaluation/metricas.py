"""Métricas de avaliação ponderadas pelo número de alunos.

Como cada linha representa muitos alunos (via `peso`), todas as métricas são
calculadas com `sample_weight`, refletindo a população real de estudantes.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def avaliar(y_true, y_prob, peso, limiar: float = 0.5) -> dict:
    """Calcula o conjunto de métricas de classificação ponderadas."""
    y_pred = (y_prob >= limiar).astype(int)
    return {
        "acuracia": accuracy_score(y_true, y_pred, sample_weight=peso),
        "precisao": precision_score(y_true, y_pred, sample_weight=peso, zero_division=0),
        "recall": recall_score(y_true, y_pred, sample_weight=peso, zero_division=0),
        "f1": f1_score(y_true, y_pred, sample_weight=peso, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob, sample_weight=peso),
        "pr_auc": average_precision_score(y_true, y_prob, sample_weight=peso),
        "brier": brier_score_loss(y_true, y_prob, sample_weight=peso),
    }


def formatar(metricas: dict) -> str:
    ordem = ["roc_auc", "pr_auc", "f1", "precisao", "recall", "acuracia", "brier"]
    return " | ".join(f"{k}={metricas[k]:.3f}" for k in ordem)
