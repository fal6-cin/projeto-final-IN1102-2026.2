# Projeto Final de Aprendizagem de Máquina (IN1102, pós-graduação, 2026.2): repositório-template

Repositório-template (`projeto-final-IN1102-2026.2`, privado) da Etapa 2 do Projeto Final da disciplina IN1102 Aprendizagem de Máquina (pós-graduação, CIn/UFPE, oferta 2026.2, release 2026.2.1), elaborado pelo Prof. Leandro Maciel Almeida. Cada grupo cria o próprio repositório a partir deste com o botão **Use this template** do GitHub, **mantém o repositório privado** e concede acesso ao professor.

O template padroniza a parte do experimento que precisa ser idêntica entre os grupos (dados, partições, métricas, baselines, orçamento de busca, protocolo de treino, validação e teste, registro dos resultados) e deixa ao grupo o trabalho substantivo: o modelo tema, a análise estatística segundo Demšar (2006), o estudo de overfitting, as agregações, os gráficos e a interpretação. O enunciado da disciplina (`entregaveis/projeto-final-IN1102-2026.2.pdf`) é a referência normativa; este README descreve como o repositório o implementa.

Resumo do experimento:

1. 30 datasets escolhidos pelo grupo entre os 38 de classificação do TabArena-v0.1 (suíte OpenML 457), 10 por regime de tamanho, antes de qualquer resultado.
2. Validação cruzada aninhada. **CV externa**: as partições oficiais das tasks do OpenML (repetição 0, 3 folds estratificados). **CV interna**: 3 folds estratificados dentro do treino de cada fold externo, onde ficam a busca de hiperparâmetros e todos os estudos.
3. Cinco sistemas: o modelo tema do grupo em duas variantes (`group_default` e `group_tuned`) e os baselines LightGBM, XGBoost e CatBoost com a configuração padrão TD do `pytabkit`.
4. Busca de hiperparâmetros com Optuna (otimização bayesiana, TPE) somente no treino de cada fold externo: 20 tentativas, CV interna de 3 folds, limite de 600 s por fold externo (verificado entre tentativas; o tempo efetivo `tune_time_s` pode exceder o limite e deve ser relatado) e subamostra estratificada de 10.000 instâncias quando o treino excede esse tamanho.
5. Três fases: desenvolvimento só no treino (`run_validation` e estudos), congelamento (`freeze`, que grava `results/protocolo.lock`, comitado no git antes do teste) e avaliação final **única** no teste de cada fold externo (`final_test`, com `results/` comitado logo depois e conferência por `check_protocol`).
6. Métricas: AUC OvO (primária), acurácia (ACC), G-Mean, entropia cruzada (CE) e tempos de busca, treino e predição.
7. Estudo de overfitting e viés-variância: lacuna treino-validação de todos os sistemas, curvas de validação de complexidade, regra de 1 erro padrão e erros por classe (matriz de confusão) nas predições de validação.
8. Estatística segundo Demšar (2006) sobre o teste (média dos 3 folds externos por dataset, N = 30): Friedman com Iman-Davenport, Holm e Bonferroni-Dunn com o modelo do grupo como controle, Nemenyi e diagrama CD como complemento (pós-testes interpretados como testes apenas se o F_F rejeitar a hipótese nula), Wilcoxon para padrão vs. com busca; intervalos por bootstrap e análise descritiva por regime.

## Status do template

Convenções:

- **Pronto**: infraestrutura implementada e coberta por testes em `tests/infra/`. Não deve ser alterada, para que os resultados dos grupos sejam comparáveis.
- **Trabalho do grupo**: o código existe com assinatura, docstring e dicas, mas o corpo levanta `NotImplementedError("TODO(grupo): ...")`. Cada item tem testes de aceitação em `tests/aceitacao/`, que aparecem como xfail até a implementação. Na correção, são executados também testes ocultos, com `AM_STRICT=1`. **Não altere as assinaturas** (nomes, parâmetros e retornos).
- **Ação do grupo**: tarefa sem código no template (decisões, execução, escrita).

| Componente | Status | Observações |
|---|---|---|
| `data/load_tabarena.py`: catálogo dos 38 datasets, validador da seleção, carregamento via OpenML com cache, partições oficiais | Pronto | `load_task` carrega o dataset inteiro; `get_folds` (treino e teste, usado só por `final_test`) e `get_train_indices` (somente treino, usado no desenvolvimento). Catálogo em `data/tabarena_v01_classificacao.csv`. |
| `src/pipeline/check_selection.py`: conferência da seleção | Pronto | `uv run python -m src.pipeline.check_selection` |
| `src/models/baselines.py`: LightGBM, XGBoost e CatBoost (TD do `pytabkit`) | Pronto | Envolvidos em `ImputedClassifier` (mediana do treino e indicador de ausência). `build_lightgbm` é usado nas curvas de `num_leaves`. |
| `src/models/wrappers.py`: `ImputedClassifier` | Pronto | Usar também com TabM, RealMLP, FT-Transformer e xRFM. |
| `src/pipeline/evaluate.py`: métricas a partir de `predict_proba` e tempos | Pronto | |
| `src/pipeline/cv_grid.py`: `stratified_subsample`, `inner_cv_splits`, `cv_grid` | Pronto | CV interna no treino de um fold externo; `cv_grid` prepara a subamostra e as partições internas e delega o laço a `validation_curve` (trabalho do grupo), medindo treino e validação de cada valor de um hiperparâmetro (estudos de sensibilidade e específico). |
| `src/pipeline/tune.py`: motor de busca com Optuna | Pronto | Orçamento padrão do enunciado; usa as mesmas partições internas de `cv_grid`. O limite de tempo é verificado entre tentativas; relate `tune_time_s`. |
| `src/pipeline/systems.py`: os cinco sistemas e `build_runner` | Pronto | Mesma construção no desenvolvimento e na avaliação final. |
| `src/pipeline/run_validation.py`: fase de desenvolvimento | Pronto | Só índices de treino dos folds externos. Grava `results/validation.csv`, `results/validation_train.csv` e `results/preds_val/` (probabilidades de validação interna); `--resume`; recusa sobrescrever `validation.csv` existente e recusa executar depois da avaliação final. |
| `src/pipeline/integrity.py`: hashes, estado do git e vestígios de avaliação | Pronto | Usado por `freeze`, `final_test`, `run_validation` e `check_protocol`. |
| `src/pipeline/freeze.py`: congelamento | Pronto | Exige repositório git. Grava `results/protocolo.lock` com o sha256 de todos os arquivos versionados em `src/**/*.py` e `data/*.py` e de `data/selecao_datasets.txt`; `--force` recongela apenas enquanto não há vestígio de avaliação final. |
| `src/pipeline/final_test.py`: avaliação final única | Pronto | Executa só em repositório git, sem alterações não comitadas em `src/`, `data/` e no lock, com o lock versionado e idêntico ao de `HEAD`, hashes conferindo e nenhum vestígio de avaliação anterior. `--resume` só completa combinações pendentes ou com erro registradas no manifesto. Grava `test.csv`, `preds/`, `test_log.csv` e `test_manifest.json`, com o commit `HEAD`. |
| `src/pipeline/check_protocol.py`: verificação pelo histórico git | Pronto | `uv run python -m src.pipeline.check_protocol`: lock comitado antes dos resultados, avaliação inicial única, hashes no commit do lock, no da avaliação e em `HEAD`, nenhuma alteração posterior sem justificativa. Usado também na correção. |
| `src/pipeline/results.py`: leitura dos resultados e regra de completude | Pronto | `load_raw` (teste), `load_validation` (desenvolvimento) e `per_dataset`, que define o N usado nos testes estatísticos. |
| `src/pipeline/split.py`: partições estratificadas para dados locais e testes | Pronto | O experimento usa as partições oficiais. |
| `src/pipeline/regime.py`: `regime_metadata`, limiares dos regimes | Pronto | |
| `src/reports/results_table.py`: `export_markdown` | Pronto | |
| `src/models/group_model.py`: `build_group_model(seed, **params)` e `search_space(trial)` | Trabalho do grupo | Teste: `tests/aceitacao/test_modelo_grupo.py` (contrato scikit-learn: rótulos de texto, atributos categóricos, categoria não vista no treino, valores ausentes, multiclasse, probabilidades válidas, predição de uma única instância, reprodutibilidade, espaço de busca). Espaço de busca justificado a priori. |
| `src/models/wrappers.py`: `TabDPTWrapper`, `ModernNCAWrapper` e `TabRWrapper` | Trabalho do grupo (apenas grupos de TabDPT-1.3, ModernNCA e TabR) | Esqueletos com dicas; validados pelo mesmo teste de contrato. Todo pré-processamento é ajustado em `fit`, só com o treino. |
| `src/pipeline/demsar.py`: postos, Friedman e Iman-Davenport, q_alpha, CD, Nemenyi, Holm, Holm contra controle, Wilcoxon (inclusive `wilcoxon_vs_control`, obrigatória), grupos e diagrama CD | Trabalho do grupo | Teste: `tests/aceitacao/test_demsar.py` (reproduz a Tabela 6 e os valores críticos da Tabela 5 de Demšar, 2006). Implementar com `numpy`, `pandas` e `scipy`. |
| `src/pipeline/bootstrap.py`: `bootstrap_ci` | Trabalho do grupo | Teste: `tests/aceitacao/test_bootstrap.py`. `bootstrap_table` e `load_predictions` estão prontos. |
| `src/pipeline/overfitting.py`: `generalization_gap`, `validation_curve` e `select_complexity` | Trabalho do grupo | Teste: `tests/aceitacao/test_overfitting.py` (lacuna orientada, curva por folds internos com pelo menos 2 folds, regra de 1 EP com desempate para o mais simples, `cv_grid` delegando a `validation_curve`). |
| `src/pipeline/class_errors.py`: `confusion_table` e `recall_per_class` | Trabalho do grupo | Teste: `tests/aceitacao/test_overfitting.py`. Aplicados às predições de validação em `results/preds_val/`; mesma API das trilhas de graduação (IF699 e CIN0209). |
| `src/pipeline/regime.py`: `aggregate_by_regime` e `rank_by_regime` | Trabalho do grupo | Teste: `tests/aceitacao/test_tabelas_regime.py`. |
| `src/reports/results_table.py`: `summary_by_model` e `per_dataset_table` | Trabalho do grupo | Teste: `tests/aceitacao/test_tabelas_regime.py`. |
| `src/reports/plots.py`: `boxplot_by_model` e `cost_vs_performance` | Trabalho do grupo | Teste: `tests/aceitacao/test_tabelas_regime.py`. |
| `src/reports/plots_overfitting.py`: `plot_validation_curve` (faixa de média ± 1 EP), `plot_generalization_gap` e `plot_confusion` | Trabalho do grupo | Teste: `tests/aceitacao/test_overfitting.py`. |
| `notebooks/01_eda.ipynb`, `02_demo_baselines.ipynb` e `03_demo_modelo_grupo.ipynb` | Pronto | Catálogo e seleção; baselines e modelo do grupo por CV interna no treino do fold externo `OUTER_FOLD` (0); busca curta e sensibilidade com `cv_grid`. Um teste de infraestrutura (`tests/infra/test_notebooks.py`) falha se os notebooks 01 a 05 referenciarem o teste. |
| `notebooks/05_overfitting.ipynb` | Trabalho do grupo | Roteiro do estudo de overfitting com sete perguntas (lacuna, curvas, regra de 1 EP e erros por classe na validação); o grupo declara o subconjunto de datasets e o hiperparâmetro de complexidade do modelo tema. |
| `notebooks/06_avaliacao_final.ipynb` | Pronto | Confere o lock, executa `final_test` (uma única vez, com `EXECUTAR = True` consciente), lê o log e o manifesto e executa `check_protocol`. |
| `notebooks/04_analise_estatistica.ipynb` | Trabalho do grupo | Roteiro com seis perguntas sobre `results/test.csv`; executado só depois da avaliação final. |
| `data/selecao_datasets.txt`: 30 task IDs | Ação do grupo | 10 pequenos, 10 médios (os 10 do catálogo) e 10 grandes; escolher pelo catálogo, antes de qualquer resultado, e justificar no relatório. |
| Escolha do extra e instalação do modelo tema | Ação do grupo | Ver [Modelos tema](#modelos-tema) e `docs/INFRAESTRUTURA.md`. |
| `.github/workflows/testes.yml` | Ação do grupo | Acrescentar `--extra <modelo>` na linha do `uv export` para que a CI instale o pacote do modelo tema; no TabPFN-3.5, cadastrar o segredo `TABPFN_TOKEN` (ver `docs/INFRAESTRUTURA.md`, seção 5). |
| Declarações a priori | Ação do grupo | Espaço de busca (com a origem das faixas; sem usar os resultados por dataset do leaderboard do TabArena), subconjunto de datasets dos estudos, grades e sentido de simplicidade, registrados em `docs/declaracoes_a_priori.md` e comitados antes de executar o notebook 03 e o `run_validation`. |
| Fase de desenvolvimento (`results/validation.csv`, `results/validation_train.csv`) | Ação do grupo | `run_validation` nos 30 datasets e 3 folds externos. |
| Estudo de sensibilidade (2 hiperparâmetros) e estudo específico do modelo | Ação do grupo (modelos com API scikit-learn nativa) | Por CV interna com `cv_grid`. Exigidos dos grupos de TabPFN-3.5, TabICL v2, TabM, RealMLP, FT-Transformer, xRFM, EBM e APLR, em compensação pelo wrapper que os grupos de TabDPT-1.3, ModernNCA e TabR implementam. |
| Congelamento e avaliação final (`protocolo.lock`, `test.csv`, `preds/`, `test_log.csv`, `test_manifest.json`) | Ação do grupo | `freeze`, commit do lock, uma única vez `final_test`, commit de `results/` e `check_protocol`. O `.gitignore` permite versionar esses arquivos, os de validação e `results/figures/`, que fazem parte da entrega (`results/preds_val/` não é versionado). |
| `model_cards/<modelo>.md` | Ação do grupo | Copiar `model_cards/TEMPLATE.md` e preencher todas as seções. |
| Relatório, slides e apresentação | Ação do grupo | Estruturas sugeridas e rubrica em `entregaveis/`. |

## Modelos tema

| # | Modelo | Extra | Ponto de entrada | Observações |
|---|---|---|---|---|
| 1 | TabPFN-3.5 | `tabpfn` | `tabpfn.TabPFNClassifier` | Variante 3.5-Fast em CPU. Pesos com licença não comercial: aceitar a licença em ux.priorlabs.ai e definir `TABPFN_TOKEN`. |
| 2 | TabICL v2 | `tabicl` | `tabicl.TabICLClassifier` | |
| 3 | TabDPT-1.3 | `tabdpt` | `src.models.wrappers.TabDPTWrapper` | Completar o esqueleto. |
| 4 | TabM | `baselines` | `pytabkit.TabM_D_Classifier` | Envolver em `ImputedClassifier`. |
| 5 | RealMLP | `baselines` | `pytabkit.RealMLP_TD_Classifier` | Envolver em `ImputedClassifier`. |
| 6 | FT-Transformer | `baselines` | `pytabkit.FTT_D_Classifier` | Envolver em `ImputedClassifier`. |
| 7 | xRFM | `baselines` | `pytabkit.XRFM_D_Classifier` | Envolver em `ImputedClassifier`. |
| 8 | ModernNCA | `modernnca` | `src.models.wrappers.ModernNCAWrapper` | Completar o esqueleto (TALENT, instalado do GitHub; exige `git`). |
| 9 | EBM | `ebm` | `interpret.glassbox.ExplainableBoostingClassifier` | |
| 10 | APLR | `aplr` | `aplr.APLRClassifier` | |
| 11 | TabR | `baselines` | `src.models.wrappers.TabRWrapper` | Completar o esqueleto (o `pytabkit.TabR_S_D_Classifier` não aceita valores ausentes nem categóricas). |

Modelos caros sem GPU: TabPFN-3.5 acima de cerca de 5.000 linhas de treino, ModernNCA, TabR e FT-Transformer nos datasets com cerca de 50.000 linhas ou mais. O protocolo ajusta cada sistema duas vezes por fold externo (desenvolvimento e avaliação final; no desenvolvimento, um ajuste por fold interno, isto é, 3 por fold externo; para `group_tuned`, a busca completa é refeita em cada fase); planeje o tempo. Para esses casos, ver a seção de Colab em `docs/INFRAESTRUTURA.md`.

## Início rápido

Pré-requisitos: `git` e [`uv`](https://docs.astral.sh/uv/). O `uv` instala o Python 3.12 automaticamente se necessário.

```bash
git clone <url-do-repositorio-do-grupo>
cd <repositorio-do-grupo>

# núcleo + baselines + ferramentas de desenvolvimento
uv sync --extra baselines

# acrescente o extra do modelo tema, por exemplo:
uv sync --extra baselines --extra ebm

# testes: infraestrutura passa; aceitação aparece como xfail até ser implementada
uv run pytest
```

Saída esperada no template recém-criado: `51 passed, 54 xfailed`. Os testes de infraestrutura criam repositórios git temporários; é preciso ter `git` instalado. À medida que o grupo implementa os componentes, os xfail passam a `passed`. Para ver o erro real de um teste de aceitação, como na correção:

```bash
AM_STRICT=1 uv run pytest tests/aceitacao        # bash
$env:AM_STRICT="1"; uv run pytest tests/aceitacao  # PowerShell
```

O marcador `model` (`uv run pytest -m model`) identifica testes adicionais do modelo tema usados na correção; nenhum teste visível do template usa esse marcador, e a execução padrão o exclui.

## Fluxo de trabalho

O fluxo segue o protocolo de treino, validação e teste do enunciado: toda decisão é tomada no treino dos folds externos, por CV interna; o teste de cada fold externo é usado uma única vez, ao final.

Os blocos de comandos usam a sintaxe do bash (Linux, macOS e Git Bash no Windows). No PowerShell do Windows 5.1, que não aceita `&&`, execute separadamente cada comando ligado por `&&`, ou use o Git Bash.

1. **Seleção dos datasets.** Explorar o catálogo em `notebooks/01_eda.ipynb`, preencher `data/selecao_datasets.txt` e conferir com `uv run python -m src.pipeline.check_selection --markdown results/tabela_datasets.md`.
2. **Modelo tema.** Implementar `build_group_model` e `search_space` em `src/models/group_model.py`, com faixas justificadas pela literatura, até `tests/aceitacao/test_modelo_grupo.py` passar. Conferir os baselines em `notebooks/02_demo_baselines.ipynb` e o modelo em `notebooks/03_demo_modelo_grupo.ipynb` (ambos por CV interna no treino do fold externo 0). Antes de executar o notebook 03, comite `src/models/group_model.py` (configuração padrão e espaço de busca) e registre as declarações a priori do passo 3; o notebook 03 é demonstrativo, só deve ser executado depois desse commit e não deve motivar mudanças no espaço de busca.
3. **Declarações a priori.** Registrar o subconjunto de datasets dos estudos, as grades de valores e o sentido de simplicidade de cada hiperparâmetro antes de olhar qualquer curva. Registre-as em `docs/declaracoes_a_priori.md` e comite o arquivo antes de executar o notebook 03 e o `run_validation`; cite o hash desse commit no relatório.
4. **Desenvolvimento (somente treino).**
   ```bash
   uv run python -m src.pipeline.run_validation            # 5 sistemas, 30 datasets, CV interna em cada fold externo
   uv run python -m src.pipeline.run_validation --resume   # retoma após interrupção
   ```
   Gera `results/validation.csv` (métricas de validação interna por dataset, fold externo, fold interno e sistema), `results/validation_train.csv` (métricas no treino interno, numa subamostra de até 2.000 instâncias) e `results/preds_val/` (probabilidades de validação interna). `load_task` carrega o dataset inteiro, mas só os índices de treino são usados: o teste não participa de nenhum ajuste nem predição antes de `final_test`. As linhas de `group_tuned` têm `optimistic = True`: os hiperparâmetros foram escolhidos nas mesmas partições internas, e o `tune_best_score` é otimista. Comite `results/validation.csv` e `results/validation_train.csv` logo após o `run_validation`.

   **`validation.csv` é apenas diagnóstico.** O treino de um fold externo contém o teste dos outros dois; por isso os sistemas, o controle `group_tuned`, o wrapper, a configuração padrão, o espaço de busca e o orçamento são fixados a priori. Depois de `run_validation`, só são permitidas correções de falha de execução (linhas com `status = error`, contrato do estimador), registradas no relatório; mudanças motivadas por desempenho não são permitidas.
5. **Estudos por CV interna.** Sensibilidade e estudo específico com `cv_grid` (`notebooks/03_demo_modelo_grupo.ipynb`, grupos com API scikit-learn nativa) e estudo de overfitting com `generalization_gap`, `validation_curve`, `select_complexity`, `confusion_table`, `recall_per_class` e os gráficos de `plots_overfitting.py` (`notebooks/05_overfitting.ipynb`, todos os grupos). Os estudos são diagnósticos: não alteram o espaço de busca, o wrapper nem a configuração padrão.
6. **Estatística.** Implementar `demsar.py`, `bootstrap_ci`, as agregações por regime, as tabelas e os gráficos até os testes de aceitação passarem (os testes usam dados sintéticos; nada disso exige o teste).
7. **Congelamento e commit do lock.**
   ```bash
   git add -A && git commit -m "Código final do desenvolvimento"   # nada pendente em src/ e data/
   uv run python -m src.pipeline.freeze
   git add results/protocolo.lock && git commit -m "Congela o protocolo"
   git push
   ```
   O `freeze` grava `results/protocolo.lock` com o sha256 de **todos** os arquivos versionados em `src/**/*.py` e `data/*.py` e de `data/selecao_datasets.txt` (quebras de linha normalizadas), a assinatura do espaço de busca, o orçamento, a semente, os sistemas, os folds, os datasets e o commit. Se for preciso corrigir algo antes do teste, `freeze --force` recongela (e o lock é comitado de novo); havendo qualquer vestígio de avaliação final, o comando recusa.
8. **Avaliação final, uma única vez, e commit imediato.**
   ```bash
   uv run python -m src.pipeline.final_test
   git add results && git commit -m "Avaliação final"
   git push
   uv run python -m src.pipeline.final_test --resume   # só para pendências ou erros, com o mesmo lock
   ```
   Ou `notebooks/06_avaliacao_final.ipynb`. O comando só executa se: (a) o diretório é um repositório git; (b) não há alterações não comitadas em `src/`, `data/` e no lock; (c) o lock está versionado e idêntico ao de `HEAD`; (d) os hashes conferem; (e) não há vestígio de avaliação anterior (`test.csv`, `test_log.csv`, `test_manifest.json`, `preds/` ou commit com esses resultados). Gera `results/test.csv`, `results/preds/`, `results/test_log.csv` e `results/test_manifest.json`, com o commit `HEAD`. Orçamento, semente, sistemas, folds e datasets vêm do lock.
9. **Conferência do protocolo.**
   ```bash
   uv run python -m src.pipeline.check_protocol
   ```
   Verifica no histórico git que o commit do lock antecede o primeiro commit com `test.csv` ou `test_log.csv`, que houve uma única avaliação inicial (retomadas só de pendências ou erros), que os hashes do lock conferem com os arquivos no commit do lock, no da avaliação e em `HEAD`, e que nenhum commit posterior altera arquivos travados sem uma linha `Justificativa:` na mensagem. Deve terminar com `APROVADO`.
10. **Análise.** Executar `notebooks/04_analise_estatistica.ipynb` sobre `results/test.csv` e responder às perguntas dos notebooks 04 e 05 no relatório. Nada do que aparece no teste retroalimenta o modelo.
11. **Model card e entregáveis.** Preencher `model_cards/<modelo>.md`, o relatório e os slides (`entregaveis/`).

As travas locais de `freeze` e `final_test` são dissuasórias: quem controla o diretório pode apagar arquivos. A evidência do protocolo é o **histórico git**: comite o lock antes do `final_test`, comite `results/` imediatamente depois e não reescreva o histórico a partir daí (sem rebase, amend ou push forçado).

Regras (ver o enunciado e `entregaveis/rubrica-IN1102.pdf`): é proibido usar os resultados por dataset do leaderboard do TabArena na seleção dos datasets e na definição do espaço de busca (a origem das faixas é discutida como ameaça à validade). Qualquer uso do teste antes do commit do lock, reexecução da avaliação final ou alteração de arquivos travados depois do teste sem justificativa anula os critérios de implementação e de interpretação da Etapa 2. A verificação usa `check_protocol`, o histórico git, `results/test_log.csv`, `results/test_manifest.json` e `results/protocolo.lock`.

## Arquivos de resultados

| Arquivo | Gerado por | Conteúdo |
|---|---|---|
| `results/validation.csv` | `run_validation` | Uma linha por (task_id, dataset, fold externo, fold interno, sistema): métricas de validação, `n_train`, `n_val`, tempos, `tune_best_score`, `optimistic`, `params`, `status`, `error`. |
| `results/validation_train.csv` | `run_validation` | Métricas no treino interno do mesmo estimador, com `n_train_eval` (até 2.000). |
| `results/preds_val/<task>_fold<f>_inner<i>_<sistema>.npz` | `run_validation` | Rótulos verdadeiros, probabilidades de validação interna e classes (base da matriz de confusão; não versionado; regenerável só antes da avaliação final). |
| `results/figures/` | notebooks 04 e 05 | Curvas de validação, lacunas, matrizes de confusão, diagramas CD, boxplots, custo vs. desempenho. |
| `results/protocolo.lock` | `freeze` | JSON com o sha256 de cada arquivo travado, espaço de busca, orçamento, semente, sistemas, folds, task IDs e commit. |
| `results/test.csv` | `final_test` | Uma linha por (task_id, dataset, fold externo, sistema): métricas de teste, tempos, `status`, `error`. |
| `results/preds/<task>_fold<f>_<sistema>.npz` | `final_test` | Rótulos verdadeiros, probabilidades de teste e classes (base do bootstrap). |
| `results/test_log.csv` | `final_test` | Colunas `timestamp, event, lock_sha256, git_head, model, status, detail`; eventos `inicio` (com `resume=True` nas retomadas), `sistema` (uma linha por combinação, `model` = `task_id|fold|sistema`), `fim` e `recusa` (tentativas recusadas, que não contam como avaliação). |
| `results/test_manifest.json` | `final_test` | `lock_sha256`, `status` de cada combinação (`pendente`, `ok`, `error`), `rows` (hash de cada linha ok de `test.csv`) e `runs` (execuções, com `git_head`). |

## Estrutura

```
.
|- README.md
|- pyproject.toml, uv.lock, .python-version, Dockerfile
|- .github/workflows/testes.yml
|- data/
|   |- load_tabarena.py
|   |- tabarena_v01_classificacao.csv
|   |- selecao_datasets.txt
|- src/
|   |- models/        baselines.py, group_model.py, wrappers.py
|   |- pipeline/      split.py, cv_grid.py, tune.py, evaluate.py, systems.py,
|   |                 run_validation.py, integrity.py, freeze.py, final_test.py,
|   |                 check_protocol.py, results.py, check_selection.py, demsar.py,
|   |                 bootstrap.py, regime.py, overfitting.py, class_errors.py
|   |- reports/       results_table.py, plots.py, plots_overfitting.py
|- notebooks/         01_eda, 02_demo_baselines, 03_demo_modelo_grupo,
|                     04_analise_estatistica, 05_overfitting, 06_avaliacao_final
|- tests/
|   |- infra/         testes da infraestrutura pronta (protocolo, avaliação final, check_protocol, notebooks)
|   |- aceitacao/     testes do trabalho do grupo (xfail até implementar)
|- model_cards/TEMPLATE.md
|- docs/INFRAESTRUTURA.md, docs/declaracoes_a_priori.md
|- entregaveis/       projeto-final-IN1102-2026.2.pdf (enunciado), relatorio-template-IN1102.pdf, slides-template-IN1102.pdf, rubrica-IN1102.pdf, fontes/
```

## Reprodutibilidade

1. Sementes fixas (padrão 42) na busca, nas partições internas, nos modelos e no bootstrap; partições oficiais do OpenML.
2. Versões exatas em `uv.lock`; Python 3.12 (`.python-version`).
3. Resultados linha a linha em `results/validation.csv`, `results/validation_train.csv` e `results/test.csv` (com status e mensagem de erro), probabilidades de teste em `results/preds/` e trilha do protocolo em `results/protocolo.lock`, `results/test_log.csv`, `results/test_manifest.json` e no histórico git (conferido por `check_protocol`).
4. Integração contínua: `.github/workflows/testes.yml` instala as versões do `uv.lock` com o `torch` para CPU e executa `ruff` e `pytest` a cada push.
5. Registrar no relatório o hash do commit congelado (o mesmo de `protocolo.lock`), os commits do lock e dos resultados, a saída de `check_protocol`, o hardware e o tempo total.

## Licenças

O código do template é distribuído sob licença MIT (arquivo `LICENSE`, titular Leandro Maciel Almeida). As dependências têm, em geral, licenças permissivas (MIT, Apache 2.0, BSD). Os pesos do TabPFN-3.5 têm licença própria, não comercial: consulte a licença de uso dos pesos antes de usá-los. Verifique a licença dos pesos do modelo tema e registre-a no model card.

## Suporte

Dúvidas técnicas: abrir issue no repositório do grupo marcando o professor (Prof. Leandro Maciel Almeida), ou contatá-lo diretamente. Consulte antes `docs/INFRAESTRUTURA.md`, seção de solução de problemas.
