# Infraestrutura e ferramentas: guia prático

Este documento descreve, passo a passo, o ambiente usado no Projeto Final da disciplina IN1102 (pós-graduação, 2026.2): instalação com `uv`, extras por modelo, testes, execução do experimento (desenvolvimento, congelamento e avaliação final), uso do Google Colab com GPU, token do TabPFN e solução de problemas. Comece pelo topo se você nunca configurou um ambiente Python para um projeto científico; consulte apenas a seção necessária caso já tenha experiência.

Sumário:

1. [Requisitos](#1-requisitos)
   - [1.1 Git e GitHub em poucos passos](#11-git-e-github-em-poucos-passos)
2. [Ambiente com uv](#2-ambiente-com-uv)
3. [Extras por modelo tema](#3-extras-por-modelo-tema)
4. [Testes: infraestrutura, aceitação, xfail e AM_STRICT](#4-testes-infraestrutura-aceitação-xfail-e-am_strict)
5. [Integração contínua (GitHub Actions)](#5-integração-contínua-github-actions)
6. [Execução do experimento: desenvolvimento, congelamento e avaliação final](#6-execução-do-experimento-desenvolvimento-congelamento-e-avaliação-final)
7. [Notebooks](#7-notebooks)
8. [Token do TabPFN-3.5](#8-token-do-tabpfn-35)
9. [Custo computacional e GPU](#9-custo-computacional-e-gpu)
10. [Plano B: Google Colab com GPU](#10-plano-b-google-colab-com-gpu)
11. [Docker (opcional)](#11-docker-opcional)
12. [Reprodutibilidade e sementes](#12-reprodutibilidade-e-sementes)
13. [Solução de problemas](#13-solução-de-problemas)


## 1. Requisitos

| Ferramenta | Versão | Como verificar |
|---|---|---|
| `git` | qualquer recente | `git --version` |
| `uv` | 0.8 ou superior | `uv --version` |
| Python | 3.12 (instalado pelo `uv` se necessário) | `uv run python --version` |
| Docker (opcional) | 24 ou superior | `docker --version` |

O projeto declara `requires-python = ">=3.11,<3.13"` e fixa 3.12 em `.python-version`. Não é preciso instalar o Python manualmente: o `uv` baixa a versão correta na primeira sincronização.

Instalação do `uv` (uma única vez):

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS e Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.1 Git e GitHub em poucos passos

Toda a evidência do protocolo está no histórico git do repositório do grupo; por isso, cada integrante precisa saber registrar e enviar o próprio trabalho. Configuração, uma única vez por computador:

1. Instale o git: no Windows, o [Git for Windows](https://git-scm.com/download/win), que inclui o Git Bash e o Git Credential Manager; no macOS, `xcode-select --install`; no Linux, o pacote `git` da distribuição.
2. Identifique-se (nome e e-mail aparecem em cada commit):
   ```bash
   git config --global user.name "Seu Nome"
   git config --global user.email "seu-email@exemplo.com"
   ```
3. Autentique-se no GitHub: no Windows, o Git Credential Manager abre o navegador no primeiro `git clone` ou `git push`; nos demais sistemas, use o GitHub CLI (`gh auth login`) ou um token pessoal (GitHub, Settings > Developer settings > Personal access tokens) no lugar da senha.

Ciclo de trabalho, na raiz do repositório (a pasta que contém `pyproject.toml`):

```bash
git clone https://github.com/<usuario>/<repositorio-do-grupo>.git   # uma vez
cd <repositorio-do-grupo>
git pull                         # traz os commits dos colegas
git status                       # mostra o que mudou
git add <arquivos>               # ou git add -A para tudo
git commit -m "Mensagem curta"   # registra (comitar = registrar em um commit)
git push                         # envia ao GitHub
```

Depois do commit do lock, **não** use `git commit --amend`, `git rebase` nem `git push --force`: reescrever o histórico elimina a evidência do protocolo (seção 6.6). Para alterar um arquivo travado depois do teste, faça um commit novo com uma linha iniciada por `Justificativa:` na mensagem, por exemplo `git commit -m "Corrige o rótulo do gráfico" -m "Justificativa: erro de digitação no eixo"`.

## 2. Ambiente com uv

Um ambiente virtual é um diretório isolado (`.venv/`) com o seu próprio Python e as suas próprias bibliotecas. O `uv` cria e mantém esse ambiente a partir de dois arquivos:

- `pyproject.toml`: dependências do núcleo (dados, métricas, busca e análise), extras opcionais por modelo e o grupo `dev` (pytest, ruff, jupyter);
- `uv.lock`: versões exatas resolvidas para todas as plataformas. Ele garante que todos os integrantes, a CI e a correção usem o mesmo ambiente. Versione-o junto com o código.

Comandos básicos:

```bash
uv sync --extra baselines        # cria .venv e instala núcleo + baselines + grupo dev
uv run pytest                    # executa um comando dentro do ambiente
uv run python -m src.pipeline.check_selection
uv run jupyter lab
```

`uv sync` remove do ambiente o que não foi pedido. Por isso, repita sempre todos os extras desejados no mesmo comando (por exemplo, `uv sync --extra baselines --extra tabicl`).

Para manter a reprodutibilidade, prefira `uv sync --frozen` depois que o `uv.lock` estiver estável: o comando falha se o lock estiver desatualizado, em vez de alterá-lo. Não atualize dependências (`uv lock --upgrade`) sem combinar com o professor, pois isso muda o ambiente de correção.

Alternativa sem `uv` (não recomendada): `python -m venv .venv`, ativar o ambiente e `pip install -e ".[baselines]"`. Nesse caso as versões do `uv.lock` não são respeitadas e o grupo deve registrar `pip freeze` no relatório.

## 3. Extras por modelo tema

| Extra | Conteúdo | Modelos tema |
|---|---|---|
| `baselines` | `torch`, `pytabkit[models]`, `lightgbm`, `xgboost`, `catboost` | baselines (todos os grupos), TabM, RealMLP, FT-Transformer, xRFM, TabR |
| `tabpfn` | `tabpfn` | TabPFN-3.5 |
| `tabicl` | `tabicl` | TabICL v2 |
| `tabdpt` | `tabdpt` | TabDPT-1.3 |
| `modernnca` | TALENT (instalado do GitHub em commit fixo; exige `git`) | ModernNCA |
| `ebm` | `interpret` | EBM |
| `aplr` | `aplr` | APLR |

Todos os grupos precisam do extra `baselines`. Exemplo para o grupo do EBM:

```bash
uv sync --extra baselines --extra ebm
```

Os modelos do `pytabkit` (TabM, RealMLP, FT-Transformer, xRFM) não aceitam valores ausentes em colunas numéricas; envolva-os em `src.models.wrappers.ImputedClassifier`, como nos baselines. Os pontos de entrada de cada modelo estão na docstring de `src/models/group_model.py` e no README.

Grupo do ModernNCA: antes de completar o `ModernNCAWrapper`, leia no TALENT instalado (`.venv/Lib/site-packages/TALENT/` no Windows; `.venv/lib/python3.12/site-packages/TALENT/` no Linux e no macOS) os arquivos `api.py` (função `build_args`), `model/method_registry.py` (função `get_method_spec` e o registro `modernNCA`), `model/methods/modernNCA.py` (treino e predição), `model/models/modernNCA.py` (arquitetura), `configs/default/modernNCA.json` e `configs/opt_space/modernNCA.json` (configuração padrão e espaço de busca publicados) e `model/lib/evaluation.py` (função usada para obter as probabilidades).

## 4. Testes: infraestrutura, aceitação, xfail e AM_STRICT

Os testes ficam em `tests/` e são executados com `uv run pytest`:

- `tests/infra/`: verificam a infraestrutura pronta (dados, métricas, modelos baselines, busca, protocolo, avaliação final, `check_protocol` e notebooks). Incluem testes "espiões" que contam as chamadas de ajuste e de predição dos estimadores e confirmam que a busca, `run_validation` (inclusive o comando completo, com `load_task` e `get_folds` substituídos) nunca usam o teste dos folds externos e que `final_test` ajusta somente no treino e faz exatamente uma predição sobre o teste por sistema e fold. Testes com repositórios git temporários verificam que `freeze` registra os hashes de todos os arquivos versionados, que `final_test` recusa executar sem o lock comitado, com alterações não comitadas, com hash divergente ou com qualquer vestígio de avaliação anterior, que `--resume` só completa pendências e erros, e que `check_protocol` aprova o fluxo correto e reprova fluxos adulterados. Uma varredura dos notebooks 01 a 05 falha se algum deles referenciar o teste. Esses testes exigem o `git` instalado. Devem passar sempre; se falharem, o ambiente está com problema (ver seção 13).
- `tests/aceitacao/`: verificam os componentes que o grupo implementa (modelo tema, `demsar.py`, `bootstrap_ci`, `overfitting.py`, `class_errors.py`, agregações por regime, tabelas e gráficos). Os testes de `cv_grid` estão aqui, porque `cv_grid` delega o laço a `validation_curve`.

Enquanto um componente levantar `NotImplementedError("TODO(grupo): ...")`, o teste correspondente aparece como **xfail** ("ainda não implementado"). Assim que o grupo implementa o componente, o teste passa a contar: se o resultado estiver correto, aparece como `passed`; se estiver errado ou levantar outro erro, aparece como `failed`.

No template recém-criado, a saída esperada é:

```
51 passed, 54 xfailed
```

A meta é chegar a todos os testes `passed`, sem nenhum xfail.

**Modo estrito.** Na correção, os testes são executados com a variável de ambiente `AM_STRICT=1`, que desliga a conversão em xfail: um componente não implementado conta como falha. Os testes visíveis são complementados por testes ocultos que exercitam casos adicionais da mesma especificação (por exemplo, empates, direção da métrica, classes raras). O teste de contrato do modelo tema já verifica, entre outros pontos, uma categoria não vista no treino e a predição de uma única instância. Para simular a correção:

```bash
AM_STRICT=1 uv run pytest                    # bash, macOS, Linux, Git Bash
$env:AM_STRICT="1"; uv run pytest            # PowerShell
Remove-Item Env:AM_STRICT                    # PowerShell: desfaz
```

Regras:

- Não altere as assinaturas das funções e classes do trabalho do grupo (nome, parâmetros, valores padrão e tipo de retorno); os testes ocultos dependem delas.
- Não altere os testes visíveis nem `tests/conftest.py`.
- Funções auxiliares privadas (prefixo `_`) podem ser criadas livremente.

Comandos úteis:

```bash
uv run pytest tests/aceitacao/test_demsar.py -x     # para no primeiro erro
uv run pytest -k wilcoxon                           # filtra por nome
uv run pytest -rx                                   # lista os motivos dos xfail
```

**Marcadores.** O `pyproject.toml` declara os marcadores `acceptance` (aplicado automaticamente aos testes de aceitação) e `model` (testes adicionais do modelo tema usados na correção). Nenhum teste visível do template usa o marcador `model`, e a execução padrão o exclui; `uv run pytest -m model` executa apenas os testes com esse marcador.

**Acentos no Git Bash (Windows).** Se a saída dos testes ou dos scripts mostrar caracteres acentuados trocados, defina a codificação do Python antes de executar: `export PYTHONIOENCODING=utf-8`.

## 5. Integração contínua (GitHub Actions)

O arquivo `.github/workflows/testes.yml` executa, a cada push e pull request, três passos:

1. **Instalar dependências (torch para CPU na CI):** cria o ambiente com `uv venv --python 3.12`, exporta as versões exatas do `uv.lock` com `uv export --frozen --no-hashes --no-emit-project --extra baselines --group dev` (descartando os pacotes de GPU `nvidia-*`, `triton` e `cuda-*`), instala-as com `uv pip install --torch-backend cpu`, que usa o `torch` para CPU, e instala o projeto com `uv pip install --no-deps -e .`;
2. **Lint:** `uv run --no-sync ruff check src data tests`;
3. **Testes (aceitação aparece como xfail até ser implementada):** `uv run --no-sync pytest`.

Como o teste de contrato do modelo tema importa o pacote do modelo, **acrescente o extra do seu modelo na linha do `uv export`**, por exemplo, para o EBM:

```yaml
          uv export --frozen --no-hashes --no-emit-project --extra baselines --extra ebm --group dev \
```

Para o TabPFN-3.5, cadastre também o token como segredo do repositório do grupo: **Settings > Secrets and variables > Actions > New repository secret**, com o nome `TABPFN_TOKEN`. O workflow já repassa esse segredo como variável de ambiente; nunca escreva o token no arquivo.

O selo verde da CI no repositório do grupo é evidência de reprodutibilidade considerada na avaliação. Execute `uv run ruff check src data tests` antes do push para evitar falhas de estilo.

## 6. Execução do experimento: desenvolvimento, congelamento e avaliação final

O experimento segue o protocolo de treino, validação e teste do enunciado (validação cruzada aninhada):

- **CV externa**: os 3 folds oficiais do OpenML (repetição 0) de cada dataset. O teste de cada fold externo é usado uma única vez, por `final_test`.
- **CV interna**: 3 folds estratificados (semente 42) dentro do treino de cada fold externo, com subamostra estratificada de até 10.000 instâncias. Nela ficam a busca e todos os estudos.

São três comandos, sempre executados na raiz do repositório, nesta ordem, intercalados com commits no git (seção 6.6), e uma conferência final com `check_protocol`.

Os blocos de comandos usam a sintaxe do bash (Linux, macOS e Git Bash no Windows). No PowerShell do Windows 5.1, que não aceita `&&`, execute separadamente cada comando ligado por `&&`, ou use o Git Bash.

### 6.1 Seleção dos datasets

Preencha `data/selecao_datasets.txt` (um task ID por linha), a partir do catálogo e antes de qualquer resultado, e confira:

```bash
uv run python -m src.pipeline.check_selection
uv run python -m src.pipeline.check_selection --markdown results/tabela_datasets.md
```

### 6.2 Desenvolvimento: `run_validation`

```bash
uv run python -m src.pipeline.run_validation             # 5 sistemas, 30 datasets, 3 folds externos
uv run python -m src.pipeline.run_validation --resume    # retoma sem refazer o que terminou
```

Para cada dataset, o comando carrega o dataset inteiro (`load_task`), mas usa **somente os índices de treino** de cada fold externo (`get_train_indices`): as linhas de teste não participam de nenhum ajuste nem de nenhuma predição antes de `final_test`. Para cada fold externo e sistema, ajusta o sistema em cada treino interno e mede as métricas na validação interna e no próprio treino interno. Para `group_tuned`, a busca Optuna é feita no treino do fold externo e os hiperparâmetros escolhidos são reavaliados nas mesmas partições internas da busca.

| Opção | Padrão | Uso |
|---|---|---|
| `--models` | os cinco sistemas | Subconjunto de `lightgbm xgboost catboost group_default group_tuned`. |
| `--task-ids` | `data/selecao_datasets.txt` | Execução parcial em datasets específicos (dispensa a validação da seleção). |
| `--folds` | `0 1 2` | Folds externos usados. |
| `--inner-folds` | 3 | Número de folds da CV interna. Mantenha o padrão. |
| `--output-dir` | `results` | Diretório de saída (útil no Colab, ver seção 10). |
| `--resume` | desligado | Pula combinações (dataset, fold externo, sistema) já concluídas com status `ok`. Sem `--resume`, o comando recusa executar se `validation.csv` já existir no diretório de saída. |
| `--n-trials`, `--tune-timeout`, `--tune-max-samples` | 20, 600, 10000 | Orçamento de busca. Mantenha o padrão; valores menores apenas para testes rápidos. O limite de tempo é verificado entre tentativas, por isso o tempo efetivo (`tune_time_s`) pode excedê-lo. |
| `--train-eval-max` | 2000 | Tamanho máximo da subamostra estratificada do treino interno em que as métricas de treino são medidas. |
| `--seed` | 42 | Semente. |
| `--skip-validation` | desligado | Dispensa a validação da seleção (apenas execuções parciais de teste). |

Saídas:

- `results/validation.csv`: uma linha por (task_id, dataset, fold externo, fold interno, sistema), com `auc_ovo`, `accuracy`, `g_mean`, `cross_entropy`, `n_train`, `n_val`, `fit_time_s`, `predict_time_s`, `tune_time_s`, `tune_best_score`, `optimistic`, `params`, `status` e `error`. As linhas de `group_tuned` têm `optimistic = True`: a validação interna foi usada na própria seleção e superestima o desempenho; `tune_best_score` é a melhor AUC OvO média da busca (também otimista) e `params` guarda os hiperparâmetros escolhidos.
- `results/validation_train.csv`: mesmas chaves, com as métricas do mesmo estimador no treino interno, medidas numa subamostra estratificada de até 2.000 instâncias (`n_train_eval`). É a base da lacuna de generalização (`generalization_gap`).
- `results/preds_val/<task>_fold<f>_inner<i>_<sistema>.npz`: rótulos verdadeiros, probabilidades na validação interna e classes, base da matriz de confusão (`confusion_table`, `recall_per_class`). Não é versionado; pode ser regenerado com `run_validation` apenas antes da avaliação final (depois dela, o comando recusa executar no diretório de resultados), por isso conclua o notebook 05 antes do `freeze`. Nunca há predições de teste nesta fase.
- `cache/openml/`: cache dos datasets e partições (variável `TABARENA_CACHE` altera o local).

Falhas não interrompem a execução: a linha recebe `status = error` e a mensagem em `error`; `--resume` refaz apenas o que não terminou com sucesso. Depois da avaliação final, o comando recusa executar.

**`validation.csv` é apenas diagnóstico.** O treino de um fold externo contém o teste dos outros dois folds externos. Por isso, os sistemas, o controle `group_tuned`, o wrapper, a configuração padrão, o espaço de busca e o orçamento são fixados a priori, e `validation.csv` não é base de decisões. Depois de `run_validation`, só são permitidas correções de **falha de execução** (linhas com `status = error`, violação do contrato do estimador, erro de instalação), registradas no relatório com a causa e a alteração; mudanças motivadas por desempenho não são permitidas.

### 6.3 Estudos por CV interna (notebooks)

Os estudos usam somente o treino de um fold externo (`get_train_indices(task_id, folds=[OUTER_FOLD])`, com `OUTER_FOLD = 0` nos notebooks 03 e 05) e as mesmas partições internas da busca (`inner_cv_splits`, `stratified_subsample`). Todo estudo exige validação cruzada com pelo menos 2 folds (`validation_curve` e `select_complexity` levantam `ValueError` com menos):

- **Sensibilidade e estudo específico** (grupos com API scikit-learn nativa): `cv_grid(build_group_model, PARAM, VALUES, X_train, y_train)` em `notebooks/03_demo_modelo_grupo.ipynb`. `cv_grid` prepara a subamostra e as partições internas e delega o laço a `validation_curve` (trabalho do grupo); devolve uma linha por (valor, fold interno), com `<métrica>_train` e `<métrica>_val`.
- **Overfitting** (todos os grupos): `notebooks/05_overfitting.ipynb`, com `generalization_gap` (a partir de `validation.csv` e `validation_train.csv`), `validation_curve` e `select_complexity` (hiperparâmetro de complexidade do modelo tema e `num_leaves` do LightGBM, num subconjunto declarado de datasets), os erros por classe com `confusion_table` e `recall_per_class` sobre `results/preds_val/` e os gráficos `plot_generalization_gap`, `plot_validation_curve` (faixa de média ± 1 EP) e `plot_confusion`.

Declare o subconjunto de datasets, as grades e o sentido de simplicidade antes de olhar os resultados. Registre essas declarações em `docs/declaracoes_a_priori.md` e comite o arquivo antes de executar o notebook 03 e o `run_validation`; o notebook 03 é demonstrativo e só deve ser executado depois desse commit. As faixas do espaço de busca vêm do artigo original, da documentação do pacote ou dos espaços publicados pelo TabArena; é proibido usar os resultados por dataset do leaderboard do TabArena na seleção dos datasets e no espaço de busca, e a origem das faixas é discutida no relatório como ameaça à validade. Os estudos são diagnósticos: não alteram o espaço de busca, o wrapper, a configuração padrão nem o lock. O treino do fold externo 0 contém as instâncias de teste dos folds 1 e 2; por isso nenhum estudo em um fold pode alterar o sistema avaliado nos demais.

### 6.4 Congelamento: `freeze` e commit do lock

```bash
git add -A && git commit -m "Código final do desenvolvimento"   # nada pendente em src/ e data/
uv run python -m src.pipeline.freeze
git add results/protocolo.lock && git commit -m "Congela o protocolo"
git push
```

Exige um repositório git. Grava `results/protocolo.lock` (JSON) com:

- `files`: sha256 (quebras de linha normalizadas, `\r\n` para `\n`) de **todos** os arquivos versionados em `src/**/*.py` e `data/*.py` e de `data/selecao_datasets.txt`;
- `search_space`: assinatura do espaço de busca (hiperparâmetros amostrados numa tentativa de semente fixa e as distribuições);
- `budget` (`n_trials`, `tune_timeout`, `tune_max_samples`), `seed`, `models`, `folds` e `task_ids`;
- `git_commit` e `git_dirty` (alterações não registradas), além de `version` e `created_at`.

| Opção | Padrão | Uso |
|---|---|---|
| `--output-dir` | `results` | Diretório onde o lock é gravado (o mesmo usado depois por `final_test`), dentro do repositório. |
| `--n-trials`, `--tune-timeout`, `--tune-max-samples`, `--seed` | 20, 600, 10000, 42 | Orçamento e semente registrados no lock e usados pela avaliação final. Mantenha o padrão. |
| `--models`, `--folds` | todos | Sistemas e folds externos registrados no lock. |
| `--force` | desligado | Substitui um lock existente (recongelar). Só funciona enquanto não há vestígio de avaliação final. |
| `--skip-validation` | desligado | Dispensa a validação da seleção (apenas testes parciais). |

O comando recusa executar fora de um repositório git, se o lock já existir sem `--force` e, mesmo com `--force`, se houver qualquer vestígio de avaliação final (`test.csv`, `test_log.csv` com evento de início, `test_manifest.json`, predições de teste em `preds/` ou commit do histórico com `test.csv` ou `test_log.csv`). Avisa se `validation.csv` não existir no diretório de saída (o caminho real é impresso), se o espaço de busca não puder ser avaliado ou se houver alterações não registradas no git. Recongelar é legítimo apenas antes do teste, para corrigir uma falha de execução encontrada na fase de desenvolvimento, e deve ser mencionado no relatório. Depois do `freeze`, **comite o lock**: o commit é a evidência de que o protocolo foi fixado antes do teste.

### 6.5 Avaliação final única: `final_test`

```bash
uv run python -m src.pipeline.final_test
git add results && git commit -m "Avaliação final"      # imediatamente depois
git push
uv run python -m src.pipeline.final_test --resume       # apenas para pendências ou erros
```

Para cada dataset e fold externo do lock, cada sistema é ajustado no treino do fold (com a busca refeita nesse treino, para `group_tuned`) e avaliado **uma vez** no teste. Orçamento, semente, sistemas, folds e datasets vêm do lock, não da linha de comando. Opções: apenas `--output-dir` (padrão `results`, onde deve estar o lock, dentro do repositório) e `--resume`.

Guardas (o comando só executa se todas forem satisfeitas):

1. o diretório é um repositório git;
2. não há alterações não comitadas em arquivos de `src/`, `data/` e `config/` (a pasta `config/` só existe em outras trilhas) nem no lock (arquivos novos `.py` também contam);
3. `results/protocolo.lock` está versionado e idêntico ao de `HEAD`;
4. os hashes do lock conferem com o repositório: mesmo conjunto de arquivos travados, mesmo conteúdo e mesma assinatura do espaço de busca;
5. não existe vestígio de avaliação anterior no diretório de saída nem em `results/` (`test.csv`, `test_log.csv` com evento de início, `test_manifest.json`, `preds/*.npz`) nem commit do histórico com `test.csv` ou `test_log.csv`.

Com `--resume`, as guardas 1 a 4 continuam valendo; o comando exige `test_manifest.json` com o mesmo sha256 do lock, recusa se alguma linha concluída de `test.csv` estiver ausente, alterada ou não registrada no manifesto (o manifesto guarda um hash por linha concluída) e completa **somente** as combinações que o manifesto registra como pendentes (interrupção) ou com erro.

Saídas:

- `results/test.csv`: uma linha por (task_id, dataset, fold externo, sistema), com `auc_ovo`, `accuracy`, `g_mean`, `cross_entropy`, `tune_time_s`, `fit_time_s`, `predict_time_s`, `total_time_s`, `status` e `error`;
- `results/preds/<task>_fold<f>_<sistema>.npz`: rótulos verdadeiros, probabilidades de teste e classes, usados pelo bootstrap;
- `results/test_log.csv`: colunas `timestamp`, `event`, `lock_sha256`, `git_head`, `model`, `status` e `detail`; eventos `inicio` (gravado antes de qualquer ajuste; `detail` traz `resume=True` nas retomadas), `sistema` (uma linha por combinação, com `model` = `task_id|fold|sistema`), `fim` e `recusa` (tentativa recusada pelas guardas, que não conta como avaliação);
- `results/test_manifest.json`: `lock_sha256`, `status` (`pendente`, `ok` ou `error` por combinação), `rows` (hash de cada linha ok de `test.csv`) e `runs` (execuções, com `git_head`, horários e combinações avaliadas).

CSV e JSON do protocolo são gravados com terminador de linha `\n` em qualquer sistema; `test.csv` tem uma linha por combinação (numa retomada, a linha com erro é substituída).

Os testes estatísticos usam apenas os datasets em que todos os sistemas comparados concluíram todos os folds externos (`per_dataset` em `src/pipeline/results.py`). Falhas persistentes reduzem N e devem ser relatadas.

### 6.6 Conferência: `check_protocol`

```bash
uv run python -m src.pipeline.check_protocol
```

Verifica, no histórico git, que: (i) o último commit que introduz ou atualiza o lock antecede o primeiro commit com `test.csv` ou `test_log.csv` (lock e resultados no mesmo commit reprovam); (ii) `test_log.csv` registra uma única avaliação inicial e `test.csv` não repete combinações concluídas (retomadas só de pendências ou erros), e o commit registrado no log fica entre o lock e os resultados; (iii) os hashes do lock conferem com os arquivos travados no commit do lock, no commit da avaliação e em `HEAD`; (iv) nenhum commit posterior à avaliação altera arquivos travados sem uma linha `Justificativa:` na mensagem (mesmo justificada, a alteração é listada para exame). Imprime um relatório com `[OK]`, `[AVISO]` e `[FALHA]` e termina com código 0 (`APROVADO`) ou 1 (`REPROVADO`). A mesma ferramenta é usada na correção.

**As travas locais são dissuasórias.** Quem controla o diretório pode apagar arquivos e repetir comandos; o histórico git não. Comite o lock antes do `final_test`, comite `results/` imediatamente depois e não reescreva o histórico a partir daí (sem rebase, amend ou push forçado).

### 6.7 Arquivos versionados

O `.gitignore` ignora `results/*`, exceto os arquivos que fazem parte da entrega:

```
!results/validation.csv
!results/validation_train.csv
!results/protocolo.lock
!results/test.csv
!results/test_log.csv
!results/test_manifest.json
!results/preds/
!results/figures/
!results/tabela_datasets.md
```

Faça commit desses arquivos ao final de cada fase: `validation.csv` e `validation_train.csv` após o desenvolvimento, o lock **antes** do `final_test` e todo o `results/` **imediatamente depois**. O cache do OpenML (`cache/`) e `results/preds_val/` não são versionados.

## 7. Notebooks

```bash
uv run jupyter lab
```

Ordem:

1. `01_eda`: catálogo e conferência da seleção;
2. `02_demo_baselines`: baselines por CV interna no treino do fold externo 0;
3. `03_demo_modelo_grupo`: modelo do grupo (após implementá-lo), busca curta de demonstração e estudo de sensibilidade com `cv_grid`, no treino do fold externo `OUTER_FOLD`;
4. `05_overfitting`: lacuna treino-validação, curvas de complexidade, regra de 1 EP e erros por classe na validação (após `run_validation` e a implementação de `overfitting.py`, `class_errors.py` e `plots_overfitting.py`);
5. `06_avaliacao_final`: confere o lock, executa `final_test` (somente com `EXECUTAR = True`, uma única vez, depois do commit do lock), lê o log e o manifesto e executa `check_protocol`;
6. `04_analise_estatistica`: roteiro sobre `results/test.csv`, executado só depois da avaliação final.

Os notebooks 02, 03 e 05 usam somente o treino de um fold externo; `tests/infra/test_notebooks.py` falha se os notebooks 01 a 05 referenciarem o teste (índices de `get_folds`, funções internas de `final_test`, `test.csv` ou predições de teste fora do notebook 04). As figuras são gravadas em `results/figures/`.

## 8. Token do TabPFN-3.5

Os pesos do TabPFN-3.5 têm licença não comercial e o download exige aceite da licença:

1. Crie uma conta em https://ux.priorlabs.ai, aceite a licença dos pesos e copie o token de acesso.
2. Defina a variável de ambiente `TABPFN_TOKEN` antes de executar:
   ```bash
   export TABPFN_TOKEN="seu-token"            # bash, macOS, Linux
   $env:TABPFN_TOKEN="seu-token"              # PowerShell (sessão atual)
   ```
   No Colab, guarde o token em **Secrets** (ícone de chave) e leia com `google.colab.userdata.get("TABPFN_TOKEN")`, atribuindo a `os.environ["TABPFN_TOKEN"]`.
3. Na primeira execução, os pesos são baixados para o cache local do pacote.

Nunca grave o token no repositório, em notebooks salvos com saída ou em arquivos versionados. Em CPU, use a variante 3.5-Fast; acima de cerca de 5.000 linhas de treino, prefira GPU.

## 9. Custo computacional e GPU

O orçamento de busca é igual para todos os grupos, mas o custo varia muito entre modelos. Referência para planejamento (CPU comum, sem GPU):

| Situação | Recomendação |
|---|---|
| Baselines, EBM, APLR, TabICL v2, xRFM, TabM e RealMLP | Executam em CPU; os datasets grandes podem levar horas no total com busca. |
| TabPFN-3.5 acima de cerca de 5.000 linhas de treino | GPU (Colab) ou subamostragem declarada do contexto. |
| TabDPT-1.3 | Viável em CPU com contexto reduzido; GPU recomendada nos grandes. |
| ModernNCA, TabR e FT-Transformer com cerca de 50.000 linhas ou mais | GPU (Colab). |
| Qualquer rede neural com espaço de busca amplo | Restringir o espaço ao que o artigo recomenda; GPU nos grandes. |

Estratégias:

- Comece pelos datasets pequenos e médios; meça o tempo por fold antes de executar os grandes.
- O protocolo ajusta cada sistema duas vezes por fold externo: na CV interna (`run_validation`) e na avaliação final (`final_test`). Planeje o tempo considerando as duas fases. No desenvolvimento, há um ajuste por fold interno (3 por fold externo); para `group_tuned`, a busca completa é refeita em cada fase.
- O regime médio, comum a todos os grupos, inclui os dois datasets de maior dimensionalidade do catálogo, Bioresponse (1.776 atributos) e hiva_agnostic (1.617 atributos categóricos, com classe minoritária de 42 instâncias), que tendem a ser os mais caros para modelos de contexto, redes neurais e modelos de vizinhança; meça o tempo neles logo no início.
- **Desenvolvimento dividido.** Use `--models` e `--task-ids` de `run_validation` para dividir a fase de desenvolvimento entre integrantes e máquinas, com `--output-dir` distinto em cada uma, e depois concatene os arquivos `validation.csv` e `validation_train.csv` em `results/` (mantendo um único cabeçalho em cada CSV). Por exemplo, em Python: `pd.concat([pd.read_csv(p) for p in partes]).to_csv("results/validation.csv", index=False, lineterminator="\n")`, em que `partes` é a lista dos `validation.csv` de cada diretório (o mesmo para `validation_train.csv`).
- **Avaliação final dividida.** De preferência, execute `final_test` inteiro em um único diretório (`results/`). Se for indispensável dividi-lo (por exemplo, o modelo do grupo no Colab com GPU e os baselines em CPU), congele **todas** as partes antes de executar qualquer uma, a partir do mesmo commit, com `freeze --models ... --output-dir results/<parte>`; comite todos os locks (`git add -f results/<parte>/protocolo.lock`); execute `final_test --output-dir results/<parte>` uma única vez em cada parte, comitando cada `results/<parte>` (`git add -f`) logo depois; confira cada parte com `check_protocol --output-dir results/<parte>`; concatene apenas os `test.csv` e as pastas `preds/` para a análise e descreva a divisão no relatório. Informe no relatório os diretórios das partes; na correção, toda pasta `results/<parte>` presente no histórico é conferida com `check_protocol --output-dir`, e partes não descritas no relatório são tratadas como reavaliação.
- Se precisar subamostrar o conjunto de contexto de um modelo de contexto em CPU (até 10.000 instâncias), implemente a regra dentro do estimador do grupo, aplique-a igualmente a todos os datasets e folds e declare-a no relatório e no model card como ameaça à validade.
- No Windows, o `torch` instalado a partir do PyPI em geral é a versão para CPU. Para usar GPU NVIDIA, a forma mais simples é o Colab ou uma máquina Linux. A CI sempre usa o `torch` para CPU (seção 5).

## 10. Plano B: Google Colab com GPU

1. Em **Ambiente de execução > Alterar o tipo de ambiente de execução**, selecione uma GPU.
2. Clone o repositório do grupo e instale o ambiente com `uv`:
   ```python
   !pip install -q uv
   !git clone https://github.com/<usuario>/<repositorio-do-grupo>.git
   %cd <repositorio-do-grupo>
   !uv sync --frozen --python 3.12 --extra baselines --extra tabpfn
   ```
   Para repositório privado, use um token pessoal do GitHub guardado em **Secrets**, nunca escrito na célula.
3. Monte o Google Drive para que os resultados sobrevivam ao fim da sessão:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
4. Fase de desenvolvimento gravando no Drive e retomando quando a sessão for interrompida:
   ```python
   !uv run python -m src.pipeline.run_validation --output-dir /content/drive/MyDrive/am-results --resume
   ```
   Na primeira execução no diretório, `--resume` apenas indica que não há nada a pular. Defina `TABARENA_CACHE=/content/drive/MyDrive/am-cache` para não baixar os datasets a cada sessão.
5. A avaliação final precisa do repositório git (o lock comitado e `results/` dentro do repositório). Clone o repositório **no Drive**, para que os arquivos sobrevivam a uma interrupção da sessão, e trabalhe nele:
   ```python
   %cd /content/drive/MyDrive
   !git clone https://github.com/<usuario>/<repositorio-do-grupo>.git
   %cd <repositorio-do-grupo>
   !git config user.name "<nome>" && git config user.email "<email>"
   !uv sync --frozen --python 3.12 --extra baselines --extra tabpfn
   !uv run python -m src.pipeline.freeze
   !git add results/protocolo.lock && git commit -m "Congela o protocolo" && git push
   !uv run python -m src.pipeline.final_test
   !git add results && git commit -m "Avaliação final" && git push
   !uv run python -m src.pipeline.check_protocol
   ```
   Copie antes, para `results/` do repositório, os arquivos `validation.csv` e `validation_train.csv` produzidos no passo 4 e comite-os. Se a sessão for interrompida durante a avaliação final, volte ao mesmo diretório do Drive e use `final_test --resume` (mesmo lock; a retomada fica registrada em `test_log.csv`). Nunca clone de novo para executar `final_test` do zero.
6. Ao final, confira no GitHub que os commits do lock e dos resultados foram enviados e que `check_protocol` termina com `APROVADO`.

Cuidados: sessões gratuitas têm limite de duração e de uso de GPU; na fase de desenvolvimento, execute por blocos de datasets (`--task-ids`); confira `nvidia-smi` antes de iniciar. Registre no relatório quais partes foram executadas no Colab e com qual GPU.

## 11. Docker (opcional)

O `Dockerfile` instala o ambiente com os extras indicados em `EXTRAS`:

```bash
docker build -t projeto-am .                                        # apenas baselines
docker build --build-arg EXTRAS="baselines ebm" -t projeto-am .     # baselines + modelo tema
docker run --rm projeto-am pytest                                   # testes
docker run --rm -it -v "$(pwd)/results":/workspace/results projeto-am bash
```

Informe explicitamente o comando a executar (por exemplo, `pytest`), como acima.

## 12. Reprodutibilidade e sementes

1. **Sementes fixas.** O padrão é `seed=42` na construção dos modelos, na busca (amostrador TPE e subamostra), nas partições internas e no bootstrap. O modelo do grupo deve repassar a semente a todas as fontes de aleatoriedade (o teste de contrato verifica a reprodutibilidade).
2. **Partições oficiais.** As partições externas vêm das tasks do OpenML (repetição 0, folds 0, 1 e 2) e são idênticas para todos os grupos; as internas são geradas por `inner_cv_splits` com semente fixa.
3. **Versões exatas.** `uv.lock` versionado; a CI instala exatamente essas versões (via `uv export --frozen`), com o `torch` para CPU.
4. **Registro.** Inclua no relatório o hash do commit congelado (o `git_commit` de `protocolo.lock`), os commits do lock e dos resultados, o sha256 do lock, o resumo de `test_log.csv`, a saída de `check_protocol`, o hardware (CPU, GPU, RAM), o tempo total e os comandos usados.
5. **Não determinismo.** Operações em GPU podem variar ligeiramente mesmo com semente fixa; documente como limitação quando ocorrer.

## 13. Solução de problemas

**`uv sync` falha ao instalar o TALENT (extra `modernnca`).**
O pacote é instalado do GitHub; verifique se o `git` está instalado e no PATH e se há acesso à internet.

**Os testes de `tests/infra/` falham logo após `uv sync`.**
Confirme que o extra `baselines` foi instalado (`uv sync --extra baselines ...`). Se o ambiente estiver inconsistente, apague `.venv/` e execute `uv sync` novamente.

**Um teste de aceitação aparece como `failed` em vez de `xfail`.**
O componente já não levanta `NotImplementedError`: o teste está avaliando a sua implementação. Leia a mensagem com `uv run pytest caminho/do/teste.py -x` e corrija.

**`ModuleNotFoundError` do pacote do modelo tema.**
O extra do modelo não foi instalado. Repita `uv sync` com todos os extras necessários. Na CI, acrescente o extra na linha do `uv export` (seção 5).

**`run_validation` recusa executar porque `results/validation.csv` já existe.**
É uma proteção contra a mistura de experimentos. Use `--resume` para continuar a execução ou `--output-dir` com outro diretório para um estudo novo.

**`validation.csv` com linhas `status=error`.**
A coluna `error` traz o tipo e a mensagem; o traceback completo é impresso no terminal. Corrija a falha de execução e execute `run_validation --resume`, que refaz apenas o que não terminou com sucesso. Nesta fase só são permitidas correções de falha de execução (erro, contrato do estimador, instalação), registradas no relatório; como `validation.csv` é apenas diagnóstico, não altere o modelo, o espaço de busca nem os sistemas por causa do desempenho observado.

**`freeze` avisa "há alterações não registradas no git".**
Faça commit do código (`git add` e `git commit`) e recongele com `freeze --force` antes do teste, para que o `git_commit` do lock identifique exatamente o código avaliado. Depois, comite o lock.

**`freeze` recusa: "Há vestígios de avaliação final".**
O teste já foi aberto (ou há resultados no histórico); o protocolo não pode mais ser recongelado. Apresente o resultado tal como obtido.

**`freeze` ou `final_test` recusa: "não é um repositório git".**
Execute os comandos na raiz do repositório clonado com `git`. O protocolo depende do histórico de commits.

**`final_test` recusa: "`results/protocolo.lock` não encontrado".**
Execute `uv run python -m src.pipeline.freeze` antes (com o mesmo `--output-dir` que será usado em `final_test`) e comite o lock.

**`final_test` recusa: "não está versionado em HEAD" ou "difere da versão comitada em HEAD".**
O lock precisa estar comitado, e o arquivo no disco precisa ser idêntico ao de `HEAD`: `git add results/protocolo.lock && git commit -m "Congela o protocolo"`.

**`final_test` recusa: "Há alterações não comitadas em arquivos do protocolo".**
A mensagem lista os arquivos de `src/`, `data/` ou o lock com alterações (ou arquivos `.py` novos não versionados). Comite-os e, se tiverem mudado depois do congelamento, recongele com `freeze --force` e comite o novo lock.

**`final_test` recusa: "O repositório diverge do protocolo congelado".**
A mensagem lista o que mudou depois do congelamento (qualquer arquivo versionado em `src/**/*.py` ou `data/*.py`, arquivos novos, `selecao_datasets.txt` ou o espaço de busca). Se o teste ainda não foi executado, restaure o arquivo (por exemplo, `git checkout -- <arquivo>`) ou, se a alteração for a correção de uma falha de execução, faça commit, recongele com `freeze --force`, comite o novo lock e mencione a correção no relatório. Se o teste já foi executado, não altere esses arquivos.

**`final_test` recusa: "A avaliação final é única e já há vestígios de uma avaliação anterior".**
A avaliação final já foi feita (a mensagem lista os arquivos ou commits encontrados). Se a execução anterior foi interrompida ou teve combinações com `status=error`, use `final_test --resume`, que completa apenas o que ficou pendente ou falhou, com o mesmo lock. Não apague arquivos para executar de novo: além de a reexecução ficar visível no histórico git, uma segunda avaliação final anula os critérios de implementação e de interpretação da Etapa 2.

**`final_test --resume` recusa: manifesto ausente ou "O lock atual difere do usado na avaliação anterior".**
O `--resume` só é aceito com `results/test_manifest.json` presente e com o mesmo `protocolo.lock` da primeira execução. Restaure o lock e o manifesto versionados (`git checkout -- results/protocolo.lock results/test_manifest.json`).

**`final_test --resume` recusa: "test.csv diverge do manifesto".**
Alguma linha concluída de `test.csv` foi removida, alterada ou acrescentada fora do `final_test`. Restaure a versão comitada (`git checkout -- results/test.csv`). Não edite `test.csv`, `test_log.csv` nem `test_manifest.json`.

**`check_protocol` reprova.**
Leia as linhas `[FALHA]`: lock não comitado ou comitado junto com os resultados, mais de uma avaliação inicial, hashes divergentes ou alteração de arquivos travados depois do teste sem `Justificativa:` na mensagem do commit. Não tente reescrever o histórico; explique a situação ao professor e no relatório.

**`test.csv` com linhas `status=error`.**
A coluna `error` traz o tipo e a mensagem. Falhas de ambiente (memória, sessão do Colab encerrada, download) podem ser completadas com `final_test --resume`; a retomada fica registrada em `test_log.csv` e deve ser explicada no relatório. Falhas persistentes reduzem N pela regra de completude e devem ser relatadas.

**Erro de valores ausentes (NaN) em TabM, RealMLP, FT-Transformer ou xRFM.**
Envolva o estimador do `pytabkit` em `ImputedClassifier`.

**Falta de memória (RAM ou VRAM) nos datasets grandes.**
Reduza o tamanho de lote ou o contexto do modelo (declarando a regra), execute no Colab com GPU ou divida a execução por dataset.

**TabPFN-3.5 recusa o download dos pesos.**
O token não foi definido na sessão atual ou a licença não foi aceita (ver seção 8).

**Erro de acesso ao OpenML ou download lento.**
Verifique a conexão. Os dados ficam em cache em `cache/openml/` após o primeiro download; a variável `TABARENA_CACHE` permite apontar para outro local.

**Caracteres acentuados trocados no Git Bash.**
Execute `export PYTHONIOENCODING=utf-8` antes dos comandos (ou acrescente a linha ao `~/.bashrc`).

**`ruff check` falha na CI.**
Execute `uv run ruff check src data tests` e, se for apenas formatação, `uv run ruff check --fix src data tests`.

**`uv sync` é lento na primeira vez.**
É esperado: a primeira sincronização baixa centenas de megabytes (PyTorch, entre outros). As seguintes usam o cache do `uv`.

**MiKTeX no Windows não compila os `.tex` de `entregaveis/fontes/`.**
Execute `pdflatex` duas vezes para resolver referências e sumário. O aviso "running on unsupported version of Windows" não impede a geração do PDF.


## Referências

- uv: https://docs.astral.sh/uv/
- pyproject.toml (PEP 621): https://peps.python.org/pep-0621/
- pytest: https://docs.pytest.org/
- GitHub Actions: https://docs.github.com/actions
- Google Colab: https://colab.research.google.com/
- Optuna: https://optuna.org/
- OpenML: https://www.openml.org/
- TabArena: https://tabarena.ai
- Docker: https://docs.docker.com/get-started/
