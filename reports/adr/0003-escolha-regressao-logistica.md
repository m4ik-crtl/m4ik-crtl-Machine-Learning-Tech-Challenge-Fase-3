# ADR 0003 — Regressão Logística como modelo final

## Contexto
Comparamos Regressão Logística, Random Forest e Hist. Gradient Boosting, todos com o mesmo
pré-processamento e ponderação por aluno.

## Decisão
Adotar a **Regressão Logística** como modelo final.

## Justificativa
- Melhor ROC-AUC na validação cruzada por município (0,699 vs 0,694 e 0,690) e melhor
  generalização no teste de 2025 (0,702 vs 0,693 e 0,678).
- Mais **interpretável** — coeficientes diretos, essenciais para decisão em política
  pública — e mais barata de treinar e manter.
- O sinal é majoritariamente estrutural/linear (IDHM-educação, meta, pobreza, região), o
  que explica os ganhos marginais ou negativos dos modelos de árvore.

## Consequências
Interpretação por coeficientes + SHAP (LinearExplainer, exato para modelo linear). Caso
futuras features tragam não-linearidades fortes, reavaliar boosting.
