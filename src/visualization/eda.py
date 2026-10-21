"""Análise exploratória — gráficos que apoiam as decisões de modelagem.

Gera as figuras em images/ e devolve um resumo numérico (também salvo em
reports/eda_resumo.json) com os achados que entram no relatório e no README.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.caminhos import IMAGES, REPORTS
from src.preprocessing.dataset import (
    FEATURES_NUMERICAS,
    carregar_gold,
)
from src.visualization.estilo import AMBER, CINZA, NAVY, REGIOES, TEAL, aplicar_estilo, salvar

ALVO = "alvo_indicador_pct"


def _dist_por_ano(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for ano, cor in zip([2023, 2024, 2025], [CINZA, TEAL, AMBER]):
        s = df.loc[df["ano"] == ano, ALVO]
        ax.hist(s, bins=40, alpha=0.6, label=f"{ano}  (média {s.mean():.1f}%)", color=cor)
    ax.set_title("Distribuição do indicador de alfabetização por município")
    ax.set_xlabel("% de crianças alfabetizadas no município")
    ax.set_ylabel("Nº de municípios")
    ax.legend()
    return salvar(fig, IMAGES / "eda_distribuicao_indicador.png")


def _por_regiao(df: pd.DataFrame):
    anos = df[df["ano"] == 2025]
    ordem = anos.groupby("regiao")[ALVO].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    dados = [anos.loc[anos["regiao"] == r, ALVO].dropna().values for r in ordem]
    bp = ax.boxplot(dados, vert=False, patch_artist=True, widths=0.6,
                    medianprops={"color": NAVY, "linewidth": 2})
    for patch, r in zip(bp["boxes"], ordem):
        patch.set_facecolor(REGIOES.get(r, TEAL))
        patch.set_alpha(0.75)
    ax.set_yticklabels(ordem)
    ax.set_title("Alfabetização por região (2025)")
    ax.set_xlabel("% de crianças alfabetizadas no município")
    return salvar(fig, IMAGES / "eda_por_regiao.png")


def _correlacao_com_alvo(df: pd.DataFrame):
    base = df[df["ano"] == 2025]
    cors = {c: base[c].corr(base[ALVO]) for c in FEATURES_NUMERICAS if c in base}
    cors = {k: v for k, v in cors.items() if pd.notna(v)}
    s = pd.Series(cors).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    cores = [TEAL if v >= 0 else "#B23A3A" for v in s.values]
    ax.barh(range(len(s)), s.values, color=cores, alpha=0.85)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index, fontsize=9)
    ax.axvline(0, color=NAVY, linewidth=0.8)
    ax.set_title("Correlação de cada variável com o indicador (2025)")
    ax.set_xlabel("Correlação de Pearson com % alfabetizados")
    return salvar(fig, IMAGES / "eda_correlacao_alvo.png")


def _idhm_scatter(df: pd.DataFrame):
    base = df[df["ano"] == 2025].dropna(subset=["idhm", ALVO])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r, cor in REGIOES.items():
        sub = base[base["regiao"] == r]
        ax.scatter(sub["idhm"], sub[ALVO], s=8, alpha=0.4, color=cor, label=r)
    # linha de tendência
    z = np.polyfit(base["idhm"], base[ALVO], 1)
    xs = np.linspace(base["idhm"].min(), base["idhm"].max(), 100)
    ax.plot(xs, np.polyval(z, xs), color=NAVY, linewidth=2, linestyle="--",
            label=f"tendência (r={base['idhm'].corr(base[ALVO]):.2f})")
    ax.set_title("IDHM × alfabetização por município (2025)")
    ax.set_xlabel("IDHM municipal (Censo 2010)")
    ax.set_ylabel("% de crianças alfabetizadas")
    ax.legend(fontsize=8, loc="lower right")
    return salvar(fig, IMAGES / "eda_idhm_scatter.png")


def _quartil_idhm(df: pd.DataFrame) -> dict:
    base = df[df["ano"] == 2025].dropna(subset=["idhm", ALVO]).copy()
    base["quartil"] = pd.qcut(base["idhm"], 4, labels=["Q1 (menor)", "Q2", "Q3", "Q4 (maior)"])
    medias = base.groupby("quartil", observed=True)[ALVO].mean()
    return {str(k): round(float(v), 1) for k, v in medias.items()}


def gerar() -> dict:
    aplicar_estilo()
    df = carregar_gold()
    _dist_por_ano(df)
    _por_regiao(df)
    _correlacao_com_alvo(df)
    _idhm_scatter(df)

    base = df[df["ano"] == 2025]
    cors = {c: float(base[c].corr(base[ALVO])) for c in FEATURES_NUMERICAS if c in base}
    cors = {k: round(v, 3) for k, v in cors.items() if pd.notna(v)}
    top = dict(sorted(cors.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6])

    resumo = {
        "n_municipios": int(df["id_municipio"].nunique()),
        "anos": sorted(int(a) for a in df["ano"].unique()),
        "media_indicador_por_ano": {
            str(int(a)): round(float(df.loc[df["ano"] == a, ALVO].mean()), 1)
            for a in sorted(df["ano"].unique())
        },
        "media_por_regiao_2025": {
            r: round(float(v), 1)
            for r, v in base.groupby("regiao")[ALVO].mean().sort_values().items()
        },
        "top_correlacoes_com_alvo": top,
        "media_por_quartil_idhm_2025": _quartil_idhm(df),
        "municipios_sem_contexto": int(df["idhm"].isna().sum()),
    }
    (REPORTS / "eda_resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumo


if __name__ == "__main__":
    import pprint
    pprint.pp(gerar())
