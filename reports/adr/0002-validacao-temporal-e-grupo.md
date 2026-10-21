# ADR 0002 — Validação temporal + GroupKFold por município

## Contexto
Com dados de 2023, 2024 e 2025 e várias linhas por município, um split aleatório vazaria
informação: o mesmo município (quase idêntico entre anos) cairia em treino e teste ao
mesmo tempo, inflando as métricas.

## Decisão
Duas camadas de proteção:
1. **Validação cruzada por grupo de município** (`GroupKFold`) dentro dos anos de treino —
   nenhum município aparece em treino e validação do mesmo fold.
2. **Teste out-of-time**: treino em 2023+2024, teste em **2025** (ano inteiro não visto).

## Consequências
- As métricas medem **generalização real** para anos e cenários futuros — exatamente o uso
  pretendido (antecipar quem não atinge metas).
- Métricas mais modestas (e honestas) do que um split aleatório daria.
