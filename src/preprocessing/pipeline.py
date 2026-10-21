"""Pré-processamento integrado ao modelo (Scikit-learn).

Todo o tratamento vive dentro de um `ColumnTransformer`, que por sua vez entra num
`Pipeline` junto do estimador. Com isso:

- a imputação e o encoding são **ajustados apenas no treino** (evita vazamento do
  conjunto de teste para o pré-processamento);
- o objeto final é um só — o mesmo pré-processamento acompanha o modelo em treino,
  validação e produção, garantindo replicabilidade.

Variáveis numéricas: imputação pela mediana + padronização.
Variáveis categóricas: imputação pela categoria mais frequente + one-hot encoding.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.dataset import FEATURES_CATEGORICAS, FEATURES_NUMERICAS


def construir_preprocessador(padronizar: bool = True) -> ColumnTransformer:
    """Monta o ColumnTransformer de pré-processamento.

    Parameters
    ----------
    padronizar:
        Se True (modelos lineares), aplica StandardScaler nas numéricas. Para
        modelos de árvore a escala é irrelevante, então pode ser desligado.
    """
    passos_num = [("imputacao", SimpleImputer(strategy="median"))]
    if padronizar:
        passos_num.append(("escala", StandardScaler()))
    ramo_numerico = Pipeline(passos_num)

    ramo_categorico = Pipeline([
        ("imputacao", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", ramo_numerico, FEATURES_NUMERICAS),
            ("cat", ramo_categorico, FEATURES_CATEGORICAS),
        ],
        remainder="drop",
    )


def nomes_features_saida(preprocessador: ColumnTransformer) -> list[str]:
    """Nomes das colunas após o pré-processamento (para importância/SHAP)."""
    nomes: list[str] = list(FEATURES_NUMERICAS)
    ohe = preprocessador.named_transformers_["cat"].named_steps["onehot"]
    nomes += list(ohe.get_feature_names_out(FEATURES_CATEGORICAS))
    return nomes
