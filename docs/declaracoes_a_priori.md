# Declarações a priori

Este arquivo registra as decisões que o grupo fixa antes de qualquer resultado experimental. Preencha todas as seções e comite o arquivo antes de executar o notebook 03 (`notebooks/03_demo_modelo_grupo.ipynb`) e o `run_validation`; cite no relatório o hash desse commit. O notebook 03 é demonstrativo e só deve ser executado depois desse commit: o que ele mostrar não motiva mudanças no wrapper, na configuração padrão nem no espaço de busca.

Depois do commit, as declarações não mudam por causa de resultados. Correções de falha de execução são feitas em commit novo, com a causa na mensagem, e explicadas no relatório; mudanças motivadas por desempenho não são permitidas.

## 1. Modelo tema, wrapper e configuração padrão

**Modelo, pacote e extra:** TabICL v2 (`tabicl.TabICLClassifier`), versão 2.2.0 (especificação `tabicl>=2.2,<3`), checkpoint oficial `tabicl-classifier-v2-20260212.ckpt`, extra `tabicl` no `pyproject.toml`.
- **Wrapper e pré-processamento:** O modelo possui API nativa compatível com o scikit-learn (`fit`, `predict_proba`, `predict`, `classes_`, `get_params`), portanto nenhum wrapper externo adicional é necessário (`build_group_model` instancia diretamente `TabICLClassifier`). O pré-processamento é estritamente encapsulado no estimador e ajustado exclusivamente dentro de `fit` sobre a partição de treino de cada split:
  - Codificação de rótulos via `y_encoder_` (`LabelEncoder`).
  - Conversão de tipos via `X_encoder_` (`TransformToNumerical`): atributos categóricos/booleanos são mapeados por `OrdinalEncoder(dtype=int64, handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)` (acomodando categorias novas no teste e valores ausentes categóricos com `-1`); atributos numéricos têm valores ausentes imputados pela **média** do treino via `SimpleImputer(strategy="mean", keep_empty_features=True)`.
  - Remoção de colunas constantes no treino via `UniqueFeatureFilter` e geração das visões do ensemble via `EnsembleGenerator`, que aplica remoção de outliers em dois estágios com *soft log-clipping* (`OutlierRemover`), transformações de normalização (`norm_methods`) e padronização por escore $z$ truncado em $[-100, 100]$ (`CustomStandardScaler`).
- **Configuração padrão (`group_default`):** Hiperparâmetros recomendados pelos autores nos artigos originais (Qu, Holzmüller, Varoquaux & Le Morvan, ICML 2025; ICML 2026) e na documentação oficial do pacote `tabicl` (`https://tabicl.readthedocs.io/`):
  - `n_estimators = 8`: tamanho de ensemble padrão adotado na avaliação do TabArena (Qu et al., 2026, Seção 6 e Apêndice L.1, Tabela L.1), que captura a maior parte do ganho de ensembling com baixo custo de inferência.
  - `norm_methods = None`: na biblioteca, resolve para `["none", "power"]`, combinando dados padronizados sem transformação não linear e com transformação Yeo-Johnson entre os membros do ensemble.
  - `feat_shuffle_method = "latin"`: permutação de atributos baseada em quadrados latinos para restaurar aproximadamente a invariância à ordem das colunas (quebrada pelo RoPE em $\text{TF}_{\text{row}}$ e pelo agrupamento circular $(0, 1, 3) \bmod m$).
  - `class_shuffle_method = "shift"`: rotação cíclica dos rótulos de classe para mitigar viés de ordem nos embeddings de classe.
  - `outlier_threshold = 4.0`: limiar de corte em unidades de desvio padrão ($|z| > 4,0$) no `OutlierRemover` de dois estágios antes do *soft log-clipping*.
  - `softmax_temperature = 0.9`: temperatura padrão $\tau = 0,9$ aplicada aos logits ($\text{softmax}(x / \tau)$) para calibração das probabilidades preditas.
  - `average_logits = True`: agregação dos membros do ensemble por média de logits antes do softmax, produzindo probabilidades mais bem calibradas (menor Log-Loss) que a média direta de probabilidades.
  - `support_many_classes = True`: ativação de *mixed-radix ensembling* no estágio $\text{TF}_{\text{col}}$ e classificação hierárquica no estágio $\text{TF}_{\text{icl}}$ para tarefas com $C > 10$ classes (Qu et al., 2026, Seção 3 e Apêndice A.3).

## 2. Espaço de busca (`search_space` em `src/models/group_model.py`)

| Hiperparâmetro | Tipo | Faixa | Escala | Fonte |
|---|---|---|---|---|
| `norm_methods` | categórico | `["none+power", "none+power+robust", "none+power+quantile_rtdl", "power", "quantile_rtdl"]` | categórica | Documentação oficial do pacote `tabicl` (`EnsembleGenerator`) e literatura RTDL/pytabkit citada em Qu et al. (2025; 2026) |
| `outlier_threshold` | real | [2.5, 6.0] | linear | Documentação oficial do pacote `tabicl` (`OutlierRemover`) e pós-processamento de caudas pesadas (Qu et al., 2025; 2026) |
| `softmax_temperature` | real | [0.5, 1.5] | log | Artigos do TabICL/TabICLv2 (Qu et al., 2025; 2026), documentação oficial do `tabicl` e literatura de calibração em PFNs (Müller et al., 2022) |
| `average_logits` | categórico | `[True, False]` | categórica | Documentação oficial do pacote `tabicl` e Apêndice G de Qu et al. (2025) (agregação por média de logits vs. probabilidades) |

As faixas não usam os resultados por dataset do leaderboard do TabArena nem os runs e avaliações publicados no OpenML para essas tasks.

### Justificativa de parcimônia dimensional e discussão como ameaça à validade

O espaço de busca foi deliberadamente concentrado nos **4 hiperparâmetros** que controlam diretamente a representação numérica de entrada ($\text{TF}_{\text{col}}$), a regularização contra valores extremos e o mecanismo de fusão probabilística do ensemble, mantendo fixos `n_estimators = 8`, `feat_shuffle_method = "latin"` e `class_shuffle_method = "shift"`:

1. **Controle da maldição da dimensionalidade sob orçamento de 20 tentativas (TPE):** O motor de busca (`src/pipeline/tune.py`) utiliza o `TPESampler` padrão do Optuna (`n_startup_trials = 10`) com orçamento estrito de 20 tentativas por fold externo. Como as 10 primeiras tentativas são amostragens aleatórias independentes e apenas as 10 seguintes são guiadas pelas densidades do Parzen Estimator, restringir o espaço a 4 dimensões (2 contínuas e 2 categóricas) preserva densidade amostral suficiente para convergência efetiva.
2. **Prevenção de estouro do limite de tempo (600 s/fold) e de memória:** No TabICLv2, o custo de inferência cresce linearmente com `n_estimators`, enquanto o ganho preditivo acima de $E = 8$ apresenta fortes retornos decrescentes ($+0,03\%$ de acurácia de $E=8$ para $E=16$ na Tabela L.1 de Qu et al., 2026, ao custo de $+67\%$ de tempo). Manter `n_estimators = 8` fixo na busca garante que nenhuma tentativa dobre artificialmente o tempo de CPU nos 3 folds internos, evitando interrupções prematuras por *timeout* (especialmente nos datasets de alta dimensionalidade do regime médio e nos datasets grandes) e reservando a avaliação da grade completa de `n_estimators` (`[1, 2, 4, 8, 16, 32]`) para a curva de complexidade (Seção 5).
3. **Eliminação de variáveis matematicamente redundantes ou sub-ótimas para AUC OvO:** A inspeção de `tabicl._sklearn.preprocessing.Shuffler` mostra que, em problemas binários ($C = 2$, que compõem 23 dos 30 datasets selecionados), existem apenas $2! = 2$ permutações possíveis de classes, de modo que `"shift"`, `"latin"` e `"random"` em `class_shuffle_method` produzem exatamente o mesmo par de permutações (efeito nulo). Analogamente, `feat_shuffle_method = "latin"` é o ótimo estrutural desenhado pelos autores para garantir cobertura ortogonal das posições rotacionais do RoPE. Por sua vez, a combinação de `average_logits` e `softmax_temperature` atua conjuntamente na ponderação de confiança entre os membros do ensemble (quando `average_logits = False` ou em problemas multiclasse) e na calibração da métrica secundária (Entropia Cruzada).
4. **Risco de vazamento de meta-conhecimento (*indirect data snooping*) e paradigma *zero-tuning*:** Embora este grupo não tenha consultado o leaderboard por dataset do TabArena nem os runs do OpenML, os autores do TabICLv2 validaram suas escolhas arquiteturais e de pré-processamento em benchmarks correlatos, além de projetarem o modelo para operar *out-of-the-box* sem tuning (Qu et al., 2026, Seções 6 e 8). Assim, há tanto o risco de alinhamento indutivo prévio do `group_default` quanto o risco de sobreajuste na validação interna de 3 folds do `group_tuned` (Cawley & Talbot, 2010).
5. **Assimetria de tratamento de valores ausentes:** O `TabICLClassifier` nativo imputa atributos numéricos ausentes pela **média** do treino sem adicionar indicadores binários de ausência (Qu et al., 2026, Seção 8), ao passo que os três baselines GBDT recebem imputação pela **mediana** acompanhada de coluna indicadora binária (`__ausente`) via `ImputedClassifier`.
6. **Imutabilidade pós-declaração:** Uma vez comitado este documento, o espaço de busca e a configuração padrão permanecem rigorosamente congelados, garantindo que nenhum ajuste a posteriori seja motivado pelos resultados obtidos na fase de validação ou diagnóstico.

## 3. Seleção dos datasets

- **Arquivo:** `data/selecao_datasets.txt`, aprovado por `uv run python -m src.pipeline.check_selection`.
- **Critérios:** Seleção foi baseada exclusivamente nos metadados do catálogo, buscando diversidade de tamanho, número de classes, tipos de atributos, presença de valores ausentes e desbalanceamento, em equilíbrio com a viabilidade computacional. Foram incluídos todos os datasets médios, preservados problemas binários e multiclasse em cada regime e incluído um dataset acima de 100 mil instâncias. Não foram consultados resultados de desempenho dos modelos.

## 4. Subconjunto de datasets dos estudos

Os estudos de sensibilidade, específico do modelo e curvas de validação de complexidade são realizados exclusivamente por validação cruzada interna (3 folds estratificados, semente 42) sobre a partição de **treino** do fold externo **`OUTER_FOLD = 0`** (com subamostra estratificada de até `MAX_SAMPLES = 10_000` instâncias quando o treino excede esse limite), utilizando o primeiro dataset selecionado em cada regime de tamanho (`STUDY_TASKS` do `notebooks/05_overfitting.ipynb`):

- **Regime pequeno (`small`):** Task `363621` — `blood-transfusion-service-center` ($n = 748$, $4$ atributos numéricos, $2$ classes, sem valores ausentes, `imbalance_ratio = 3.20`).
- **Regime médio (`medium`):** Task `363700` — `seismic-bumps` ($n = 2.584$, $15$ atributos sendo $20\%$ categóricos, $2$ classes, sem valores ausentes, `imbalance_ratio = 14.20`).
- **Regime grande (`large`):** Task `363628` — `customer_satisfaction_in_airline` ($n = 129.880$, $21$ atributos sendo $76\%$ categóricos, $2$ classes, com valores ausentes, `imbalance_ratio = 1.21`).

## 5. Grades dos estudos e sentido de simplicidade

| Estudo | Hiperparâmetro | Grade de valores | Sentido de simplicidade |
|---|---|---|---|
| Curva de complexidade do TabICL v2 | `n_estimators` | `[1, 2, 4, 8, 16, 32]` | menor é mais simples (`SIMPLER_IS_LOWER = True`) |
| Estudo de sensibilidade 1 | `softmax_temperature` | `[0.5, 0.7, 0.9, 1.2, 1.5]` | maior é mais simples (`SIMPLER_IS_LOWER = False`: maior temperatura suaviza a distribuição predita e reduz a nitidez extrema) |
| Estudo de sensibilidade 2 | `outlier_threshold` | `[2.5, 3.0, 4.0, 5.0, 6.0]` | menor é mais simples (`SIMPLER_IS_LOWER = True`: menor limiar restringe mais fortemente valores extremos) |
| Estudo específico do modelo | `norm_methods` (e `feat_shuffle_method`) | `["none", "power", "quantile_rtdl", "robust", "none+power"]` (e `["none", "shift", "random", "latin"]`) | categórico (`"none"` é a variante mais simples, sem transformação não linear nem permutação) |

## 6. Métricas

- **Métrica primária:** AUC OvO (`auc_ovo`).
- **Métrica secundária:** Entropia cruzada (`cross_entropy` / Log-Loss), escolhida a priori por ser uma regra de pontuação estritamente própria (*strictly proper scoring rule*) que avalia diretamente a qualidade da calibração das probabilidades preditas pelo modelo in-context (Qu et al., 2025; 2026).

## 7. Regras de subamostragem ou de contexto (se houver)

Nenhuma regra adicional customizada no estimador (`TabICLClassifier` utiliza integralmente o conjunto de treino recebido como contexto, apoiado pelo mecanismo QASSMax para generalização em contextos longos). Aplica-se exclusivamente a regra padrão da infraestrutura do protocolo: subamostragem estratificada de até $10.000$ instâncias (`stratified_subsample`, semente 42) apenas durante a busca de hiperparâmetros (`tune.py`) e os estudos de CV interna (`cv_grid.py`) quando o treino do fold externo excede $10.000$ linhas; o ajuste final de avaliação em cada fold utiliza o treino completo.
