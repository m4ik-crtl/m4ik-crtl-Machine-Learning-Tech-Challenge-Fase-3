# Dados — camada Gold (origem Fase 2)

Os dados deste projeto vêm da **camada Gold** construída no Tech Challenge da Fase 2
(pipeline medalhão sobre o Indicador Criança Alfabetizada do INEP/MEC).

Para tornar a Fase 3 **reprodutível de forma independente**, os snapshots das tabelas
Gold usados na modelagem são versionados aqui em `data/gold/`, em formato Parquet:

- `features_ml_municipio.parquet` — feature store sem vazamento (16.713 município-ano).
- `contagens_alunos_municipio.parquet` — contagens de alunos avaliados / alfabetizados por
  município-ano, usadas para montar o alvo binário por aluno e os pesos.

## Procedência

- **Fonte:** camada Gold da Fase 2 (formato Delta Lake).
- **Natureza dos dados:** indicadores, metas e contexto socioeconômico por município-ano.
  As fontes IBGE, Atlas do Desenvolvimento Humano (Censo 2010) e série INEP/MEC são reais;
  o grão municipal e os microdados de aluno são simulação calibrada (cada UF reproduz o
  valor publicado), conforme declarado na Fase 2.
- **Sem vazamento:** `features_ml_municipio` exclui qualquer variável derivada do alvo.

## Dicionário resumido de `features_ml_municipio`

| Coluna | Descrição |
|---|---|
| `ano` | Ano de referência |
| `id_municipio` | Código IBGE (7 dígitos) |
| `nome_municipio`, `sigla_uf`, `regiao` | Identificação territorial |
| `alvo_indicador_pct` | Indicador de alfabetização do município no ano t (base do alvo) |
| `indicador_ano_anterior_pct` | Indicador em t-1 (feature sem vazamento) |
| `matriculas_ano_anterior`, `variacao_ano_anterior_pp` | Histórico sem vazamento |
| `meta_pactuada_pct` | Meta conhecida ex-ante |
| `idhm`, `idhm_educacao`, `idhm_renda`, `renda_per_capita`, `indice_gini` | Desenvolvimento/renda |
| `pct_pobres`, `pct_criancas_pobres` | Pobreza |
| `taxa_analfabetismo_15mais`, `pct_6a14_fora_escola`, `pct_criancas_dom_sem_fund`, `pct_6a14_fund_sem_atraso` | Educação/contexto |
| `pct_agua_encanada`, `pct_energia_eletrica` | Infraestrutura domiciliar |
| `populacao_total`, `porte_municipio`, `capital` | Porte e papel do município |
