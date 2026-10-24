# Guia de contribuição

Checklist antes de abrir um Pull Request neste repositório:

- [ ] `pytest -q` passa (quando houver testes) e o notebook executa sem erro.
- [ ] Nenhum **caminho absoluto de máquina**, e-mail pessoal ou nome de usuário em
      código, docs ou notebooks — caminhos sempre relativos (`src/caminhos.py`).
- [ ] Nenhum segredo em texto puro (usar `.env`, que é ignorado pelo Git).
- [ ] Documentação de preparação pessoal (roteiro de vídeo, checklist de entrega) fica
      **fora** do repositório.
- [ ] `ruff check .` sem erros.
- [ ] Dados pesados ficam em `data/` com procedência documentada em `data/README.md`.

Fluxo: branch de feature → commit → Pull Request → revisão → merge na `main`.
