"""Treino, validação e seleção do modelo final.

Estratégia de validação (anti-vazamento em duas camadas):

1. Validação cruzada por **grupo de município** (GroupKFold) dentro dos anos de
   treino: nenhum município aparece ao mesmo tempo em treino e validação de um
   fold, então o modelo não "decora" municípios.
2. Teste **out-of-time**: depois de escolhido pela CV, o modelo é reavaliado em
   2025, um ano inteiro que não entrou no treino.

Tudo ponderado pelo número de alunos (`sample_weight`).
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from src.caminhos import MODELS, REPORTS
from src.evaluation.metricas import avaliar, formatar
from src.modeling.modelos import modelos_candidatos
from src.preprocessing.dataset import montar


def validar_cruzado(modelo, X, y, peso, grupos, n_folds: int = 5) -> dict:
    """GroupKFold ponderado; devolve média e desvio do ROC-AUC entre folds."""
    gkf = GroupKFold(n_splits=n_folds)
    aucs = []
    for tr_idx, va_idx in gkf.split(X, y, groups=grupos):
        m = _clonar(modelo)
        m.fit(X.iloc[tr_idx], y[tr_idx], clf__sample_weight=peso[tr_idx])
        prob = m.predict_proba(X.iloc[va_idx])[:, 1]
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(y[va_idx], prob, sample_weight=peso[va_idx]))
    return {"auc_cv_media": float(np.mean(aucs)), "auc_cv_desvio": float(np.std(aucs)),
            "auc_por_fold": [float(a) for a in aucs]}


def _clonar(modelo):
    from sklearn.base import clone
    return clone(modelo)


def executar() -> dict:
    dados = montar()
    X_tr, y_tr, w_tr = dados["X_treino"], dados["y_treino"], dados["peso_treino"]
    X_te, y_te, w_te = dados["X_teste"], dados["y_teste"], dados["peso_teste"]
    grupos = dados["grupos_treino"]

    resultados = {}
    modelos = modelos_candidatos()
    for nome, modelo in modelos.items():
        print(f"\n### {nome}")
        cv = validar_cruzado(modelo, X_tr, y_tr, w_tr, grupos)
        print(f"  CV ROC-AUC: {cv['auc_cv_media']:.3f} ± {cv['auc_cv_desvio']:.3f}")
        modelo.fit(X_tr, y_tr, clf__sample_weight=w_tr)
        prob_te = modelo.predict_proba(X_te)[:, 1]
        met = avaliar(y_te, prob_te, w_te)
        print(f"  Teste 2025: {formatar(met)}")
        resultados[nome] = {"cv": cv, "teste_2025": met}

    melhor = max(resultados, key=lambda k: resultados[k]["cv"]["auc_cv_media"])
    print(f"\n==> Melhor modelo (CV): {melhor}")

    # Reajusta o melhor em todo o treino e persiste.
    modelo_final = _clonar(modelos[melhor])
    modelo_final.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    joblib.dump(modelo_final, MODELS / "modelo_final.joblib")

    relatorio = {"melhor_modelo": melhor, "resultados": resultados}
    (REPORTS / "metricas_modelos.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    return relatorio


if __name__ == "__main__":
    executar()
