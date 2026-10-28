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
  - `n_estimators = 8`: tamanho de ensemble padrão da biblioteca. O estudo agregado no TALENT apresentado no Apêndice L.1 de Qu et al. (2026) relata retornos decrescentes acima desse valor; esse resultado não constitui uma previsão de custo ou desempenho nos datasets do grupo.
  - `norm_methods = None`: na biblioteca, resolve para `["none", "power"]`, combinando dados padronizados sem transformação não linear e com transformação Yeo-Johnson entre os membros do ensemble.
  - `feat_shuffle_method = "latin"`: permutação de atributos baseada em quadrados latinos para restaurar aproximadamente a invariância à ordem das colunas (quebrada pelo RoPE em $\text{TF}_{\text{row}}$ e pelo agrupamento circular $(0, 1, 3) \bmod m$).
  - `class_shuffle_method = "shift"`: rotação cíclica dos rótulos de classe para mitigar viés de ordem nos embeddings de classe.
  - `outlier_threshold = 4.0`: limiar de corte em unidades de desvio padrão ($|z| > 4,0$) no `OutlierRemover` de dois estágios antes do *soft log-clipping*.
  - `softmax_temperature = 0.9`: temperatura padrão $\tau = 0,9$ aplicada aos logits ($\text{softmax}(x / \tau)$) para calibração das probabilidades preditas.
  - `average_logits = True`: agregação dos membros do ensemble por média de logits antes do softmax, como opção padrão da biblioteca. Seu efeito sobre a qualidade das probabilidades será avaliado, sem assumir superioridade em todos os datasets.
  - `support_many_classes = True`: ativação de *mixed-radix ensembling* no estágio $\text{TF}_{\text{col}}$ e classificação hierárquica no estágio $\text{TF}_{\text{icl}}$ para tarefas com $C > 10$ classes (Qu et al., 2026, Seção 3 e Apêndice A.3).

## 2. Espaço de busca (`search_space` em `src/models/group_model.py`)

| Hiperparâmetro | Tipo | Faixa | Escala | Fonte e origem da escolha |
|---|---|---|---|---|
| `norm_methods` | categórico | `["none+power", "none+power+robust", "none+power+quantile_rtdl", "power", "quantile_rtdl"]` | categórica | Opções suportadas pela API oficial (`EnsembleGenerator`); combinações selecionadas a priori pelo grupo para comparar alternativas de representação, incluindo o padrão `none+power` |
| `outlier_threshold` | real | [2.5, 6.0] | linear | API oficial (`OutlierRemover`) para a semântica e o padrão 4.0; limites escolhidos pelo grupo para explorar tratamento mais e menos restritivo de extremos |
| `softmax_temperature` | real | [0.5, 1.5] | log | API oficial para a semântica e o padrão 0.9; limites e escala log escolhidos pelo grupo para comparar temperaturas positivas abaixo e acima do padrão |
| `average_logits` | categórico | `[True, False]` | categórica | API oficial: comparação a priori entre as duas opções suportadas, média de logits (padrão) e média de probabilidades |

As combinações, os limites e as escalas acima são decisões do grupo, não intervalos exatos prescritos pelos autores. A documentação fundamenta o significado dos parâmetros: https://tabicl.readthedocs.io/en/latest/api.html. Artigo v2: https://arxiv.org/abs/2602.11139v2. As faixas não usam os resultados por dataset do leaderboard do TabArena nem os runs e avaliações publicados no OpenML para essas tasks.

### Justificativa de parcimônia dimensional e discussão como ameaça à validade

O espaço de busca foi deliberadamente concentrado nos **4 hiperparâmetros** que controlam diretamente a representação numérica de entrada ($\text{TF}_{\text{col}}$), a regularização contra valores extremos e o mecanismo de fusão probabilística do ensemble, mantendo fixos `n_estimators = 8`, `feat_shuffle_method = "latin"` e `class_shuffle_method = "shift"`:

1. **Busca restrita sob orçamento limitado:** Mantemos quatro hiperparâmetros (dois contínuos e dois categóricos) para concentrar a exploração sob o orçamento do enunciado: até 20 tentativas por fold externo e limite de 600 s, verificado entre tentativas. O TPE instalado usa `n_startup_trials = 10`; se o limite de tempo for atingido antes, pode não haver tentativas guiadas pelo TPE. Restringir a busca é uma escolha de planejamento, não uma garantia de cobertura suficiente, convergência ou encontro do ótimo. Relataremos a quantidade de tentativas concluídas e o tempo efetivo.
2. **Preservação do tamanho padrão do ensemble:** Manter `n_estimators = 8` evita acrescentar essa dimensão à busca e busca limitar o custo computacional, sem garantir ausência de falta de memória ou duração inferior a 600 s. A tentativa em andamento termina mesmo após o limite, e o tempo efetivo pode excedê-lo. O Apêndice L.1 de Qu et al. (2026) apresenta retornos decrescentes no TALENT; os tempos publicados não são previsões para nosso hardware. A grade `[1, 2, 4, 8, 16, 32]` permanece reservada ao estudo de complexidade.
3. **Preservação das permutações padrão e interpretação das probabilidades:** Mantemos `feat_shuffle_method = "latin"` e `class_shuffle_method = "shift"` por serem os padrões dos autores e para limitar a dimensionalidade da busca. Há 24 problemas binários e 6 multiclasse na seleção. Embora existam apenas duas permutações de rótulos em um problema binário, isso não prova equivalência das previsões entre estratégias: a ordenação e a composição do ensemble também participam da construção das visões. Com `average_logits=True`, temperatura positiva preserva a ordenação dos escores binários em aritmética exata, podendo mudar entropia cruzada sem mudar AUC. Com média de probabilidades ou em multiclasse, os efeitos podem ser diferentes. Portanto, não tratamos temperatura como necessariamente inútil na busca nem assumimos superioridade universal de uma estratégia.
4. **Risco de vazamento de meta-conhecimento (*indirect data snooping*) e paradigma *zero-tuning*:** Embora este grupo não tenha consultado o leaderboard por dataset do TabArena nem os runs do OpenML, os autores do TabICLv2 validaram suas escolhas arquiteturais e de pré-processamento em benchmarks correlatos, além de projetarem o modelo para operar *out-of-the-box* sem tuning (Qu et al., 2026, Seções 6 e 8). Assim, há tanto o risco de alinhamento indutivo prévio do `group_default` quanto o risco de sobreajuste na validação interna de 3 folds do `group_tuned` (Cawley & Talbot, 2010).
5. **Assimetria de tratamento de valores ausentes:** O `TabICLClassifier` nativo imputa atributos numéricos ausentes pela **média** do treino sem adicionar indicadores binários de ausência (Qu et al., 2026, Seção 8), ao passo que os três baselines GBDT recebem imputação pela **mediana** acompanhada de coluna indicadora binária (`__ausente`) via `ImputedClassifier`.
6. **Imutabilidade pós-declaração:** Uma vez comitado este documento, o espaço de busca e a configuração padrão permanecem rigorosamente congelados, garantindo que nenhum ajuste a posteriori seja motivado pelos resultados obtidos na fase de validação ou diagnóstico.

## 3. Seleção dos datasets

- **Arquivo:** `data/selecao_datasets.txt`, aprovado por `uv run python -m src.pipeline.check_selection`.
- **Critérios:** Seleção foi baseada exclusivamente nos metadados do catálogo, buscando diversidade de tamanho, número de classes, tipos de atributos, presença de valores ausentes e desbalanceamento, em equilíbrio com a viabilidade computacional. Foram incluídos todos os datasets médios, preservados problemas binários e multiclasse em cada regime e incluído um dataset acima de 100 mil instâncias. Não foram consultados resultados de desempenho dos modelos.

## 4. Subconjunto de datasets dos estudos

Os estudos de sensibilidade, específico do modelo e curvas de validação de complexidade são realizados exclusivamente por validação cruzada interna (3 folds estratificados, semente 42) sobre a partição de **treino** do fold externo **`OUTER_FOLD = 0`** (com subamostra estratificada de até `MAX_SAMPLES = 10_000` instâncias quando o treino excede esse limite), utilizando os IDs explicitamente fixados em `STUDY_TASKS` nos notebooks 03 e 05, sem depender da ordem do arquivo de seleção:

- **Regime pequeno (`small`):** Task `363621` — `blood-transfusion-service-center` ($n = 748$, $4$ atributos numéricos, $2$ classes, sem valores ausentes, `imbalance_ratio = 3.20`).
- **Regime médio (`medium`):** Task `363704` — `students_dropout_and_academic_success` (n = 4.424, 36 atributos, sendo 17 categóricos, 3 classes, sem valores ausentes; menor classe com 794 exemplos e razão maior/menor classe de aproximadamente 2,78).
- **Regime grande (`large`):** Task `363628` — `customer_satisfaction_in_airline` ($n = 129.880$, $21$ atributos sendo $76\%$ categóricos, $2$ classes, com valores ausentes, `imbalance_ratio = 1.21`).

### Justificativa da revisão do subconjunto

Após revisão coletiva das declarações, o representante médio dos estudos passa de `seismic-bumps` (363700) para `students_dropout_and_academic_success` (363704). A escolha se baseia apenas no catálogo: acrescenta um problema de três classes ao subconjunto e maior representação da menor classe (794 exemplos, contra 170). Isso amplia a cobertura da análise por classe e permite observar o estudo de temperatura também em multiclasse, sem garantir antecipadamente menor variabilidade ou melhor desempenho. Os representantes pequeno e grande permanecem iguais. A seleção principal de 30 datasets, o espaço de busca, seus intervalos, as grades e o orçamento não são alterados. A busca do experimento principal continua sendo realizada separadamente nos 30 datasets; esta troca afeta apenas o subconjunto dos estudos detalhados.

## 5. Grades dos estudos e sentido de simplicidade

| Estudo | Hiperparâmetro | Grade de valores | Sentido de simplicidade |
|---|---|---|---|
| Curva de complexidade do TabICL v2 | `n_estimators` | `[1, 2, 4, 8, 16, 32]` | menor é mais simples (`SIMPLER_IS_LOWER = True`) |
| Estudo de sensibilidade 1 | `softmax_temperature` | `[0.5, 0.7, 0.9, 1.2, 1.5]` | maior é mais simples (`SIMPLER_IS_LOWER = False`: maior temperatura suaviza a distribuição predita e reduz a nitidez extrema) |
| Estudo de sensibilidade 2 | `outlier_threshold` | `[2.5, 3.0, 4.0, 5.0, 6.0]` | menor é mais simples (`SIMPLER_IS_LOWER = True`: menor limiar restringe mais fortemente valores extremos) |
| Estudo específico do modelo | `norm_methods` (e `feat_shuffle_method`) | `["none", "power", "quantile_rtdl", "robust", "none+power"]` (e `["none", "shift", "random", "latin"]`) | categórico (`"none"` é a variante mais simples, sem transformação não linear nem permutação) |

## 6. Métricas

- **Métrica primária:** AUC OvO (`auc_ovo`).
- **Métrica secundária:** Entropia cruzada (`cross_entropy` / Log-Loss), escolhida a priori por ser uma regra de pontuação estritamente própria (*strictly proper scoring rule*) que avalia a qualidade das probabilidades preditas e penaliza previsões incorretas muito confiantes; não é uma medida exclusiva de calibração do modelo in-context (Qu et al., 2025; 2026).

## 7. Regras de subamostragem ou de contexto (se houver)

Nenhuma regra adicional customizada no estimador (`TabICLClassifier` utiliza integralmente o conjunto de treino recebido como contexto, apoiado pelo mecanismo QASSMax para generalização em contextos longos). Aplica-se exclusivamente a regra padrão da infraestrutura do protocolo: subamostragem estratificada de até $10.000$ instâncias (`stratified_subsample`, semente 42) apenas durante a busca de hiperparâmetros (`tune.py`) e os estudos de CV interna (`cv_grid.py`) quando o treino do fold externo excede $10.000$ linhas; o ajuste final de avaliação em cada fold utiliza o treino completo.
