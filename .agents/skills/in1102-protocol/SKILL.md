---
name: in1102-protocol
description: >-
  Guia operacional passo a passo do protocolo experimental da disciplina IN1102 (2026.2),
  cobrindo seleção de datasets, validação interna, estudos de overfitting (regra de 1 EP),
  congelamento (freeze), teste final único, verificação do protocolo (check_protocol) e
  análise estatística de Demšar (2006).
---

# Runbook do Protocolo Experimental IN1102 (2026.2)

Este skill orienta a execução ordenada e segura do pipeline experimental de ponta a ponta.

## Fluxo em 10 Passos

### 1. Seleção dos Datasets
- Escolha 30 datasets do catálogo TabArena-v0.1 (`data/tabarena_v01_classificacao.csv`):
  - 10 Pequenos ($n < 2.500$)
  - 10 Médios ($2.500 \le n < 10.000$) - exatamente os 10 do catálogo
  - 10 Grandes ($n \ge 10.000$)
- Grave os task IDs em `data/selecao_datasets.txt`.
- Valide a seleção:
  ```powershell
  uv run python -m src.pipeline.check_selection --markdown results/tabela_datasets.md
  ```

### 2. Modelo Tema e Espaço de Busca
- Implemente `build_group_model` e `search_space` em `src/models/group_model.py`.
- Se aplicável (TabDPT, ModernNCA, TabR), complete o wrapper em `src/models/wrappers.py`.
- Valide o contrato:
  ```powershell
  uv run pytest tests/aceitacao/test_modelo_grupo.py
  ```

### 3. Declarações a Priori
- Registre em `docs/declaracoes_a_priori.md` o espaço de busca, o subconjunto de datasets para estudos, as grades de hiperparâmetros, sentido de simplicidade e a métrica primária (AUC OvO).
- Comite este arquivo antes de rodar qualquer validação ou notebook demonstrativo.

### 4. Fase de Desenvolvimento (Validação Interna)
- Executa os 5 sistemas nos 30 datasets com CV interna (3 folds) dentro do treino dos 3 folds externos:
  ```powershell
  uv run python -m src.pipeline.run_validation
  # Caso seja interrompido:
  uv run python -m src.pipeline.run_validation --resume
  ```
- Comite `results/validation.csv` e `results/validation_train.csv`.
- Nota: `results/preds_val/` é local (não versionado) e necessário para o cálculo das matrizes de confusão.

### 5. Estudos por CV Interna (Notebooks 03 e 05)
- **Sensibilidade e Estudo Específico:** Executado via `cv_grid` no fold externo 0 para modelos com API nativa scikit-learn.
- **Overfitting e Viés-Variância:**
  - `generalization_gap(raw, raw_train, metric, higher_is_better)`
  - `validation_curve(factory, param, values, X, y, folds, seed)`
  - `select_complexity(curve, metric, rule="1se", ...)` aplicando a regra de 1 erro padrão ($m(v) \ge m(v^\star) - EP(v^\star)$)
  - `confusion_table` e `recall_per_class` nas predições de validação interna.

### 6. Estatística e Relatórios
- Implemente e valide `src/pipeline/demsar.py`, `src/pipeline/bootstrap.py`, `src/pipeline/regime.py`, `src/reports/results_table.py` e `src/reports/plots*.py`:
  ```powershell
  $env:AM_STRICT="1"; uv run pytest tests/aceitacao
  ```

### 7. Congelamento do Protocolo (`freeze`)
- Verifique que `git status` está limpo (tudo comitado em `src/` e `data/`).
- Execute o freeze:
  ```powershell
  uv run python -m src.pipeline.freeze
  git add results/protocolo.lock
  git commit -m "Congela o protocolo"
  git push
  ```

### 8. Avaliação Final Única (`final_test`)
- Executa avaliação nos conjuntos de teste de cada fold externo:
  ```powershell
  uv run python -m src.pipeline.final_test
  git add results
  git commit -m "Avaliação final"
  git push
  ```
- Se houver falhas pontuais de execução, retome com:
  ```powershell
  uv run python -m src.pipeline.final_test --resume
  ```

### 9. Conferência do Protocolo (`check_protocol`)
- Verifique se a integridade git e os hashes do lock estão aprovados:
  ```powershell
  uv run python -m src.pipeline.check_protocol
  ```
  O comando deve terminar obrigatoriamente com a palavra `APROVADO`.

### 10. Análise Final e Entregáveis
- Execute `notebooks/04_analise_estatistica.ipynb` sobre `results/test.csv`.
- Demšar (2006):
  1. Friedman com correção de Iman-Davenport ($F_F$).
  2. Se $F_F$ rejeitar $H_0$: Holm contra o controle (`group_tuned`), diagrama de Bonferroni-Dunn e teste de Nemenyi / CD. Se não rejeitar, apresente apenas como análise descritiva.
  3. Wilcoxon signed-ranks entre `group_default` e `group_tuned`.
- Complete o `model_cards/<modelo>.md`, relatório em PDF e slides segundo os templates.

