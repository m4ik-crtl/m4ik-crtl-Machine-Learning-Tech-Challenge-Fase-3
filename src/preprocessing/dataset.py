"""Construção do dataset supervisionado a partir da camada Gold.

Problema (enunciado Fase 3): prever se uma criança do 2º ano será **alfabetizada
ou não** (classificação binária), usando variáveis educacionais, territoriais e
socioeconômicas da camada Gold da Fase 2.

Desenho do alvo
---------------
A feature store Gold (`features_ml_municipio`) é municipal: cada linha é um
município-ano e o indicador é a *taxa* de alfabetização (contínua). O enunciado,
porém, pede o rótulo binário por **aluno**. Reconciliamos os dois sem perder
fidelidade nem explodir o volume:

- Para cada município-ano há N alunos avaliados, dos quais `alunos_alfabetizados`
  são positivos (y=1) e `alunos_nao_alfabetizados` negativos (y=0). Todos herdam
  as features municipais (que são justamente territoriais/socioeconômicas).
- Em vez de materializar os ~7,9 milhões de alunos, representamos cada
  município-ano por **duas linhas** (y=1 e y=0) com `peso` = contagem de alunos
  naquela célula. Treinar com `sample_weight=peso` é matematicamente idêntico a
  treinar sobre a expansão completa, 237x menor. As métricas são ponderadas pelo
  peso, refletindo a população real de alunos.

Anti-vazamento (data leakage)
-----------------------------
1. A feature store já exclui qualquer variável derivada do alvo (o indicador do
   próprio ano, contagens do próprio ano etc.). Só entram variáveis conhecidas
   *ex-ante*: indicador do ano anterior (t-1), meta pactuada, IDHM e contexto
   socioeconômico (Censo), porte, região.
2. Validação **temporal**: treino em 2023+2024, teste em 2025 — o modelo prevê um
   ano que não viu. É o cenário real de "antecipar quem pode não atingir metas".
3. A imputação e o encoding são ajustados **só no treino** (ficam dentro do
   Pipeline do Scikit-learn), nunca no conjunto de teste.
"""
from __future__ import annotations

import pandas as pd

from src.caminhos import GOLD

# Colunas que NUNCA podem virar feature (vazamento direto do alvo do mesmo ano).
COLUNAS_VAZAMENTO = [
    "alvo_indicador_pct",       # é o próprio alvo (taxa do ano t)
    "indicador_pct",            # idem, vindo das contagens
    "matriculas_avaliadas",     # contagem do ano t
    "alunos_alfabetizados",     # contagem do ano t (numerador do alvo)
    "alunos_nao_alfabetizados",  # contagem do ano t
    "meta_pct",                 # duplicata de meta_pactuada_pct
    "origem_indicador",
]

# Identificadores (mantidos para análise, fora das features de treino).
COLUNAS_ID = ["id_municipio", "nome_municipio", "ano"]

# Features categóricas conhecidas ex-ante.
FEATURES_CATEGORICAS = ["sigla_uf", "regiao", "porte_municipio"]

# Features numéricas conhecidas ex-ante (sem vazamento).
# Obs.: `variacao_ano_anterior_pp` foi deliberadamente excluída — ela depende de
# t-2, que não existe para os anos de treino (2023/2024), ficando disponível só em
# 2025. Mantê-la criaria descasamento de disponibilidade entre treino e teste.
FEATURES_NUMERICAS = [
    "indicador_ano_anterior_pct",
    "matriculas_ano_anterior",
    "meta_pactuada_pct",
    "idhm",
    "idhm_educacao",
    "idhm_renda",
    "renda_per_capita",
    "indice_gini",
    "pct_pobres",
    "pct_criancas_pobres",
    "taxa_analfabetismo_15mais",
    "pct_6a14_fora_escola",
    "pct_criancas_dom_sem_fund",
    "pct_6a14_fund_sem_atraso",
    "pct_agua_encanada",
    "pct_energia_eletrica",
    "populacao_total",
    "capital",
]

FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS

ANOS_TREINO = [2023, 2024]
ANO_TESTE = 2025


def carregar_gold() -> pd.DataFrame:
    """Lê os snapshots Gold e junta features + contagens de alunos por município-ano."""
    feat = pd.read_parquet(GOLD / "features_ml_municipio.parquet")
    cont = pd.read_parquet(GOLD / "contagens_alunos_municipio.parquet")
    df = feat.merge(
        cont[["id_municipio", "ano", "alunos_alfabetizados", "alunos_nao_alfabetizados"]],
        on=["id_municipio", "ano"],
        how="inner",
    )
    return df


def construir_dataset_binario(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma município-ano em duas linhas (y=1 / y=0) com peso = nº de alunos.

    O resultado tem, por município-ano, uma linha positiva e uma negativa, cada
    uma carregando as features municipais e um `peso` igual à contagem de alunos
    naquela classe. Linhas de peso zero são descartadas.
    """
    cols = COLUNAS_ID + FEATURES
    base = df[cols].copy()

    pos = base.copy()
    pos["y"] = 1
    pos["peso"] = df["alunos_alfabetizados"].to_numpy()

    neg = base.copy()
    neg["y"] = 0
    neg["peso"] = df["alunos_nao_alfabetizados"].to_numpy()

    out = pd.concat([pos, neg], ignore_index=True)
    out = out[out["peso"] > 0].reset_index(drop=True)
    return out


def separar_treino_teste(ds: pd.DataFrame):
    """Split temporal: treino em 2023+2024, teste em 2025 (validação out-of-time)."""
    treino = ds[ds["ano"].isin(ANOS_TREINO)].reset_index(drop=True)
    teste = ds[ds["ano"] == ANO_TESTE].reset_index(drop=True)
    return treino, teste


def montar() -> dict:
    """Pipeline de dados completo: lê a Gold e devolve treino/teste prontos."""
    df = carregar_gold()
    ds = construir_dataset_binario(df)
    treino, teste = separar_treino_teste(ds)

    def _xyw(parte: pd.DataFrame):
        return parte[FEATURES], parte["y"].to_numpy(), parte["peso"].to_numpy()

    X_tr, y_tr, w_tr = _xyw(treino)
    X_te, y_te, w_te = _xyw(teste)
    return {
        "df_municipio": df,
        "treino": treino,
        "teste": teste,
        "X_treino": X_tr, "y_treino": y_tr, "peso_treino": w_tr,
        "X_teste": X_te, "y_teste": y_te, "peso_teste": w_te,
        "grupos_treino": treino["id_municipio"].to_numpy(),
    }


if __name__ == "__main__":
    dados = montar()
    tr, te = dados["treino"], dados["teste"]
    print(f"Treino: {tr.shape[0]} linhas ponderadas | {int(dados['peso_treino'].sum()):,} alunos")
    print(f"Teste:  {te.shape[0]} linhas ponderadas | {int(dados['peso_teste'].sum()):,} alunos")
    taxa_tr = dados["y_treino"] @ dados["peso_treino"] / dados["peso_treino"].sum()
    taxa_te = dados["y_teste"] @ dados["peso_teste"] / dados["peso_teste"].sum()
    print(f"Taxa de alfabetização — treino: {taxa_tr:.1%} | teste: {taxa_te:.1%}")
    print(f"Features: {len(FEATURES)} ({len(FEATURES_NUMERICAS)} num + {len(FEATURES_CATEGORICAS)} cat)")
