"""Caminhos do projeto — 100% relativos à raiz do repositório.

Nenhum caminho absoluto de máquina deve existir em qualquer arquivo versionado.
Tudo é resolvido a partir da localização deste arquivo, então o projeto roda igual
em qualquer máquina que clone o repositório.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

DATA = RAIZ / "data"
GOLD = DATA / "gold"
NOTEBOOKS = RAIZ / "notebooks"
REPORTS = RAIZ / "reports"
IMAGES = RAIZ / "images"
MODELS = RAIZ / "models"

for _p in (DATA, GOLD, REPORTS, IMAGES, MODELS):
    _p.mkdir(parents=True, exist_ok=True)
