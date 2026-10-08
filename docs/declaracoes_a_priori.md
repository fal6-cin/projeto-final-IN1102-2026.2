# Declarações a priori

Este arquivo registra as decisões que o grupo fixa antes de qualquer resultado experimental. Preencha todas as seções e comite o arquivo antes de executar o notebook 03 (`notebooks/03_demo_modelo_grupo.ipynb`) e o `run_validation`; cite no relatório o hash desse commit. O notebook 03 é demonstrativo e só deve ser executado depois desse commit: o que ele mostrar não motiva mudanças no wrapper, na configuração padrão nem no espaço de busca.

Depois do commit, as declarações não mudam por causa de resultados. Correções de falha de execução são feitas em commit novo, com a causa na mensagem, e explicadas no relatório; mudanças motivadas por desempenho não são permitidas.

## 1. Modelo tema, wrapper e configuração padrão

- **Modelo, pacote e extra:** <nome, versão, pacote e extra do `pyproject.toml`>
- **Wrapper e pré-processamento:** <wrapper usado, se houver; pré-processamento, sempre ajustado dentro de `fit`, só com o treino>
- **Configuração padrão (`group_default`):** <hiperparâmetros e fonte (artigo original ou documentação do pacote)>

## 2. Espaço de busca (`search_space` em `src/models/group_model.py`)

| Hiperparâmetro | Tipo | Faixa | Escala | Fonte |
|---|---|---|---|---|
| <...> | <inteiro, real ou categórico> | <...> | <linear ou log> | <artigo original, documentação do pacote ou espaço publicado pelo TabArena> |

As faixas não usam os resultados por dataset do leaderboard do TabArena nem os runs e avaliações publicados no OpenML para essas tasks.

## 3. Seleção dos datasets

- **Arquivo:** `data/selecao_datasets.txt`, aprovado por `uv run python -m src.pipeline.check_selection`.
- **Critérios:** <resumo dos critérios usados, baseados apenas no catálogo; a justificativa completa vai no relatório>

## 4. Subconjunto de datasets dos estudos

<task IDs e nomes dos datasets usados nos estudos por CV interna (sugestão: um por regime de tamanho) e o fold externo usado (`OUTER_FOLD`)>

## 5. Grades dos estudos e sentido de simplicidade

| Estudo | Hiperparâmetro | Grade de valores | Sentido de simplicidade |
|---|---|---|---|
| Curva de complexidade do modelo tema | <...> | <...> | <menor é mais simples ou maior é mais simples> |
| Curva de complexidade do LightGBM | `num_leaves` | <...> | menor é mais simples |
| Estudo de sensibilidade (quando exigido) | <...> | <...> | <...> |
| Estudo específico do modelo (quando exigido) | <...> | <...> | <...> |

## 6. Métricas

- **Métrica primária:** AUC OvO.
- **Métrica secundária:** <métrica usada na repetição da análise>
- **Nível de significância:** alfa = 0,05.

## 7. Regras de subamostragem ou de contexto (se houver)

<regra, aplicada igualmente a todos os datasets e folds, ou "nenhuma">
