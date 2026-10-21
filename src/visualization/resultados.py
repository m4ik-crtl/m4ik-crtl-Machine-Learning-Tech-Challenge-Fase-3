"""Gráficos de resultado do modelo: ROC, calibração, matriz de confusão e risco.

Todos ponderados pelo número de alunos (sample_weight).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve

from src.caminhos import IMAGES, REPORTS
from src.modeling.modelos import modelos_candidatos
from src.preprocessing.dataset import montar
from src.visualization.estilo import AMBER, CINZA, CORAL, NAVY, TEAL, aplicar_estilo, salvar

CORES_MODELO = {
    "regressao_logistica": TEAL,
    "random_forest": AMBER,
    "hist_gradient_boosting": CINZA,
}
ROTULO = {
    "regressao_logistica": "Regressão Logística",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist. Gradient Boosting",
}


def _roc(modelos_fit, dados):
    from sklearn.metrics import roc_auc_score
    X_te, y_te, w_te = dados["X_teste"], dados["y_teste"], dados["peso_teste"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for nome, m in modelos_fit.items():
        prob = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, prob, sample_weight=w_te)
        auc = roc_auc_score(y_te, prob, sample_weight=w_te)
        ax.plot(fpr, tpr, color=CORES_MODELO[nome], linewidth=2.2,
                label=f"{ROTULO[nome]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#B0B7C3", linewidth=1)
    ax.set_title("Curva ROC — teste out-of-time (2025)")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.legend(loc="lower right", fontsize=9)
    return salvar(fig, IMAGES / "resultado_roc.png")


def _calibracao(modelo, dados):
    X_te, y_te, w_te = dados["X_teste"], dados["y_teste"], dados["peso_teste"]
    prob = modelo.predict_proba(X_te)[:, 1]
    frac, med = calibration_curve(y_te, prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#B0B7C3", label="perfeita")
    ax.plot(med, frac, "o-", color=TEAL, linewidth=2, label="modelo final")
    ax.set_title("Calibração do modelo final (2025)")
    ax.set_xlabel("Probabilidade prevista de alfabetização")
    ax.set_ylabel("Fração realmente alfabetizada")
    ax.legend(loc="upper left")
    return salvar(fig, IMAGES / "resultado_calibracao.png")


def _matriz(modelo, dados):
    X_te, y_te, w_te = dados["X_teste"], dados["y_teste"], dados["peso_teste"]
    pred = (modelo.predict_proba(X_te)[:, 1] >= 0.5).astype(int)
    cm = confusion_matrix(y_te, pred, sample_weight=w_te)
    cm_pct = cm / cm.sum() * 100
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(cm, cmap="BuGn")
    rot = ["Não alfabetizado", "Alfabetizado"]
    ax.set_xticks([0, 1], rot); ax.set_yticks([0, 1], rot)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão (alunos, 2025)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]/1000:.0f} mil\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else NAVY, fontsize=11)
    return salvar(fig, IMAGES / "resultado_matriz_confusao.png")


def _risco_municipios(modelo, dados):
    """Municípios de maior risco em 2025: menor probabilidade prevista de alfabetização."""
    teste = dados["teste"]
    pos = teste[teste["y"] == 1].copy()  # uma linha por município (features idênticas p/ y=0/1)
    X = pos[dados["X_teste"].columns]
    pos["prob_alfabetizacao"] = modelo.predict_proba(X)[:, 1]
    pos["risco"] = 1 - pos["prob_alfabetizacao"]
    cols = ["id_municipio", "nome_municipio", "sigla_uf", "prob_alfabetizacao", "risco"]
    ranking = pos.sort_values("risco", ascending=False)[cols].reset_index(drop=True)
    ranking.to_csv(REPORTS / "municipios_risco_2025.csv", index=False)

    top = ranking.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    rotulos = top["nome_municipio"] + " / " + top["sigla_uf"]
    ax.barh(range(len(top)), top["risco"] * 100, color=CORAL, alpha=0.85)
    ax.set_yticks(range(len(top)), rotulos, fontsize=9)
    ax.set_xlabel("Risco previsto (% de não alfabetização)")
    ax.set_title("15 municípios de maior risco educacional (2025)")
    return salvar(fig, IMAGES / "resultado_risco_municipios.png"), ranking


def gerar() -> dict:
    aplicar_estilo()
    dados = montar()
    modelos = modelos_candidatos()
    for m in modelos.values():
        m.fit(dados["X_treino"], dados["y_treino"], clf__sample_weight=dados["peso_treino"])

    _roc(modelos, dados)
    final = modelos["regressao_logistica"]
    _calibracao(final, dados)
    _matriz(final, dados)
    _, ranking = _risco_municipios(final, dados)

    insights = {
        "n_municipios_risco_alto_2025": int((ranking["risco"] > 0.5).sum()),
        "top5_municipios_risco": ranking.head(5)[
            ["nome_municipio", "sigla_uf", "prob_alfabetizacao"]
        ].round(3).to_dict("records"),
    }
    (REPORTS / "insights_risco.json").write_text(
        json.dumps(insights, indent=2, ensure_ascii=False), encoding="utf-8")
    return insights


if __name__ == "__main__":
    import pprint
    pprint.pp(gerar())
