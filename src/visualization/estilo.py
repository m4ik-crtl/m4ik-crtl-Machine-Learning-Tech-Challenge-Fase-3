"""Estilo visual compartilhado dos gráficos (paleta e defaults do matplotlib)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Paleta do projeto — educação/confiança (verde-azulado + âmbar de destaque).
TEAL = "#2E7D6B"
NAVY = "#16233F"
AMBER = "#E8A33D"
CORAL = "#B23A3A"
CINZA = "#5B6B8C"
REGIOES = {
    "Norte": "#4C8CBF",
    "Nordeste": "#B23A3A",
    "Centro-Oeste": "#E8A33D",
    "Sudeste": "#2E7D6B",
    "Sul": "#6B4E9E",
}


def aplicar_estilo():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#88909F",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def salvar(fig, caminho):
    fig.savefig(caminho)
    plt.close(fig)
    return caminho
