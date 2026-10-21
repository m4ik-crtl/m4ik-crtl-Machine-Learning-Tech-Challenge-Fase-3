"""Interpretabilidade do modelo final: importância por permutação, coeficientes e SHAP.

Responde à exigência do enunciado de "identificar quais variáveis possuem maior
influência na predição", usando Feature Importance e SHAP Values.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from src.caminhos import IMAGES, REPORTS
from src.modeling.modelos import SEMENTE, modelos_candidatos
from src.preprocessing.dataset import FEATURES, FEATURES_NUMERICAS
from src.preprocessing.pipeline import nomes_features_saida
from src.visualization.estilo import CORAL, NAVY, TEAL, aplicar_estilo, salvar


def _permutacao(modelo, dados):
    X_te, y_te, w_te = dados["X_teste"], dados["y_teste"], dados["peso_teste"]
    r = permutation_importance(
        modelo, X_te, y_te, scoring="roc_auc", n_repeats=10,
        random_state=SEMENTE, sample_weight=w_te, n_jobs=-1,
    )
    imp = pd.Series(r.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(imp)), imp.values, color=TEAL, alpha=0.85,
            xerr=r.importances_std[np.argsort(r.importances_mean)], ecolor="#9AA9C9")
    ax.set_yticks(range(len(imp)), imp.index, fontsize=9)
    ax.set_title("Importância por permutação (queda no ROC-AUC)")
    ax.set_xlabel("Queda média no ROC-AUC ao embaralhar a variável")
    salvar(fig, IMAGES / "importancia_permutacao.png")
    return imp


def _coeficientes(modelo):
    prep = modelo.named_steps["prep"]
    clf = modelo.named_steps["clf"]
    nomes = nomes_features_saida(prep)
    coefs = pd.Series(clf.coef_[0], index=nomes).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    cores = [TEAL if v >= 0 else CORAL for v in coefs.values]
    ax.barh(range(len(coefs)), coefs.values, color=cores, alpha=0.85)
    ax.set_yticks(range(len(coefs)), coefs.index, fontsize=8)
    ax.axvline(0, color=NAVY, linewidth=0.8)
    ax.set_title("Coeficientes da Regressão Logística (features padronizadas)")
    ax.set_xlabel("Efeito no log-odds de alfabetização")
    salvar(fig, IMAGES / "importancia_coeficientes.png")
    return coefs


def _shap(modelo, dados, n_amostra=2000):
    prep = modelo.named_steps["prep"]
    clf = modelo.named_steps["clf"]
    nomes = nomes_features_saida(prep)

    X_tr_t = prep.transform(dados["X_treino"])
    X_te_t = prep.transform(dados["X_teste"])
    rng = np.random.default_rng(SEMENTE)
    fundo = X_tr_t[rng.choice(len(X_tr_t), size=min(500, len(X_tr_t)), replace=False)]
    amostra_idx = rng.choice(len(X_te_t), size=min(n_amostra, len(X_te_t)), replace=False)
    X_amostra = X_te_t[amostra_idx]

    explainer = shap.LinearExplainer(clf, fundo, feature_names=nomes)
    sv = explainer(X_amostra)

    # Beeswarm
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv.values, X_amostra, feature_names=nomes, show=False, max_display=14)
    fig = plt.gcf(); fig.suptitle("SHAP — distribuição do impacto por variável", y=1.02,
                                  fontsize=13, fontweight="bold")
    salvar(fig, IMAGES / "shap_beeswarm.png")

    # Barra (média |SHAP|)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(sv.values, X_amostra, feature_names=nomes, plot_type="bar",
                      show=False, max_display=14, color=TEAL)
    fig = plt.gcf(); fig.suptitle("SHAP — importância média (|impacto|)", y=1.02,
                                  fontsize=13, fontweight="bold")
    salvar(fig, IMAGES / "shap_barra.png")

    mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=nomes).sort_values(ascending=False)
    return mean_abs


def gerar() -> dict:
    aplicar_estilo()
    from src.preprocessing.dataset import montar
    dados = montar()
    modelo = modelos_candidatos()["regressao_logistica"]
    modelo.fit(dados["X_treino"], dados["y_treino"], clf__sample_weight=dados["peso_treino"])

    imp = _permutacao(modelo, dados)
    coefs = _coeficientes(modelo)
    shap_imp = _shap(modelo, dados)

    resumo = {
        "top_permutacao": imp.sort_values(ascending=False).head(8).round(4).to_dict(),
        "top_coeficientes_abs": coefs.abs().sort_values(ascending=False).head(8).round(3).to_dict(),
        "top_shap": shap_imp.head(8).round(3).to_dict(),
    }
    (REPORTS / "interpretabilidade.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumo


if __name__ == "__main__":
    import pprint
    pprint.pp(gerar())
