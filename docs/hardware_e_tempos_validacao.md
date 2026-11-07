# Registro de hardware e tempos — validação interna

**Projeto:** IN1102 — TabICL v2 e baselines LightGBM, XGBoost e CatBoost  
**Fase documentada:** desenvolvimento e validação interna (`src.pipeline.run_validation`)  
**Fonte dos tempos:** `results/validation.csv`, commit [`87d84d1`](https://github.com/leticialevleitao/projetoAM20262Q2/commit/87d84d1b28945a04be2317ae52514832961f6302), idêntico ao arquivo consolidado auditado.  
**Cobertura:** 30 datasets × 3 folds externos × 3 folds internos × 5 sistemas = **1.350 avaliações válidas**.  
**Situação do teste externo:** ainda não incluído neste registro; o tempo de `final_test` será documentado separadamente após sua execução.

## 1. Hardware e ambiente de execução

| Item | Registro / evidência |
|---|---|
| Plataforma de execução | Google Colab, com GPU NVIDIA Tesla T4. |
| GPU — memória | **15.360 MiB de VRAM**, conforme registro de `nvidia-smi`. |
| Driver NVIDIA | **580.82.07**, conforme registro de `nvidia-smi`. |
| Python | **3.12**, versão configurada em `.python-version` e no notebook de retomada. |
| Dependências | Arquivo `uv.lock` versionado no repositório. |

A configuração da GPU e do driver foi extraída dos registros de execução do Colab; os tempos apresentados nas seções seguintes foram obtidos dos CSVs de validação.

## 2. Definição e fonte dos tempos

Os valores abaixo são **tempos medidos pelo pipeline, em segundos**, não duração de sessões do Google Colab:

- **`fit_time_s`:** tempo da chamada `estimator.fit()` de cada fold interno.
- **`predict_time_s`:** tempo de `estimator.predict_proba()` na validação de cada fold interno; não inclui a predição adicional usada nas métricas de treino.
- **`tune_time_s`:** duração da busca Optuna para o `group_tuned`, realizada **uma vez por combinação de dataset e fold externo**. O pipeline repete esse mesmo valor nas três linhas internas correspondentes. Para totalizar buscas, cada combinação é contada **uma única vez**.

## 3. Tempos por sistema

Medianas calculadas a partir das **270 avaliações internas por sistema**. A mediana da busca do TabICL otimizado considera **90 buscas distintas**, não 270 buscas independentes.

| Sistema | Avaliações | Treino — mediana (s) | Predição — mediana (s) | Busca Optuna — mediana (s) |
|---|---:|---:|---:|---:|
| LightGBM | 270 | 0,640 | 0,036 | — |
| XGBoost | 270 | 1,359 | 0,027 | — |
| CatBoost | 270 | 2,536 | 0,026 | — |
| TabICL padrão | 270 | 0,952 | 2,144 | — |
| TabICL otimizado | 270 | 0,725 | 2,102 | 166,123 |

**Totais acumulados das operações medidas:**

- Ajustes nos 1.350 folds internos: **3469,651 s** (aproximadamente **57,8 min**).
- Predições de validação nos 1.350 folds internos: **2367,258 s** (aproximadamente **39,5 min**).
- Buscas Optuna em 90 combinações distintas de dataset e fold externo: **17292,053 s** (aproximadamente **4,80 h**).

Esses totais são **somas de durações instrumentadas**, não o tempo cronológico total de execução do projeto. Não incluem integralmente carregamento e preparação de dados, criação de ambientes, download de checkpoints, cálculo de métricas, predições adicionais no treino, gravação de arquivos, reinícios de sessão ou tentativas incompletas. Também não permitem deduzir tempo de uso de GPU ou custo computacional total.

## 4. Orçamento de busca Optuna

O protocolo prevê até **20 tentativas** e um limite de **600 s por fold externo**, verificado **entre** tentativas. No `group_tuned`, os tempos registrados nas 90 buscas foram:

| Indicador | Tempo |
|---|---:|
| Mínimo | 40,607 s |
| Mediana | 166,123 s |
| Máximo | 679,555 s |
| Buscas com `tune_time_s > 600 s` | **6/90** |

As seis buscas acima de 600 s ocorreram no Bioresponse (`363620`, folds 0–2) e no hiva_agnostic (`363677`, folds 0–2), com tempos entre **625,185 s** e **679,555 s**. Isso não demonstra, por si só, violação do orçamento: o limite é conferido entre tentativas, e a última tentativa pode ultrapassá-lo.

## 5. Comparabilidade e rastreabilidade

Os resultados foram consolidados a partir de diferentes sessões, com recuperações seletivas após falhas de execução. Assim, os tempos documentados caracterizam **as avaliações efetivamente registradas**, não medições controladas em hardware rigorosamente idêntico. Ajustes de memória e de pré-processamento aplicados durante a recuperação de falhas, além de variações entre sessões do Colab, podem afetar a comparação de tempos.

O arquivo `results/validation.csv`, referenciado pelo commit indicado no cabeçalho, contém os registros utilizados neste levantamento.

**Situação:** hardware e tempos de busca, ajuste e predição registrados para a fase de validação interna. A avaliação final será documentada separadamente após sua execução.
