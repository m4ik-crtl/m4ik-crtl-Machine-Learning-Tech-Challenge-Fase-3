# ADR 0001 — Alvo binário por aluno com pesos, a partir de dados municipais

## Contexto
O enunciado pede prever se **um aluno** será alfabetizado (classificação binária). A
feature store Gold da Fase 2 é **municipal**, com alvo contínuo (taxa de alfabetização do
município). Precisávamos reconciliar o grão municipal com o rótulo binário por aluno.

## Decisão
Representar cada município-ano por **duas linhas** — uma positiva (y=1) e uma negativa
(y=0) — carregando as features municipais, com `peso` igual ao número de alunos
alfabetizados e não alfabetizados. O treino usa `sample_weight=peso` e todas as métricas
são ponderadas.

## Consequências
- **Fiel ao enunciado**: o alvo é binário por aluno; as métricas (AUC, F1) são de
  classificação e refletem a população real de ~7,9 milhões de alunos.
- **Eficiente**: 33.426 linhas em vez de 7,9 milhões (237× menor), sem perda de informação
  — treinar com pesos é idêntico a treinar sobre a expansão completa.
- **Limitação assumida**: alunos do mesmo município recebem a mesma predição (risco
  contextual, não individual). Documentado nas limitações.
