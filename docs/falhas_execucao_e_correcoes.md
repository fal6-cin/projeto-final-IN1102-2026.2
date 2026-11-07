# Registro de falhas de execução e correções

**Experimento:** TabICL v2 (`group_default` e `group_tuned`) e baselines LightGBM, XGBoost e CatBoost

**Fase:** desenvolvimento e validação interna (`src.pipeline.run_validation`)

**Situação:** validação interna consolidada, com **1.350/1.350 avaliações válidas**; **nenhuma avaliação pendente**.

**Ambiente de execução:** Google Colab, Python 3, GPU NVIDIA T4.

**Commits das correções:** [`58c2425`](https://github.com/leticialevleitao/projetoAM20262Q2/commit/58c242557b87beab2af72d47dbd068515776c2d8), [`2cec625`](https://github.com/leticialevleitao/projetoAM20262Q2/commit/2cec625de52d37337f0cafaf80c71d7cd9802cdb) e [`d8a4d73`](https://github.com/leticialevleitao/projetoAM20262Q2/commit/d8a4d73ea85001c690e417df88a36d36c96aaba0)

**Commit dos CSVs consolidados:** [`87d84d1b28945a04be2317ae52514832961f6302`](https://github.com/leticialevleitao/projetoAM20262Q2/commit/87d84d1b28945a04be2317ae52514832961f6302).

> Este documento registra falhas observadas na execução e suas correções. Essas intervenções não foram motivadas por resultados de desempenho. As correções de código estão registradas nos commits identificados acima. A avaliação final (`freeze`/`final_test`) constitui etapa distinta e não é objeto deste registro.

## 1. Escopo e restrições metodológicas

O protocolo a priori define **30 datasets**, **cinco sistemas**, **três folds externos** e **três folds internos**, totalizando `30 × 5 × 3 × 3 = 1.350` avaliações da validação interna. A busca do `group_tuned` utiliza Optuna, com **até 20 tentativas**, **orçamento de 600 segundos por fold externo** (verificado entre tentativas), subamostra estratificada de até **10.000 instâncias** e semente padrão **42**. O teste do fold externo não é utilizado na busca; a avaliação final é reservada a `src.pipeline.final_test`, após o congelamento por `src.pipeline.freeze`.

Em conformidade com `docs/declaracoes_a_priori.md`, não foram propostas alterações nos datasets selecionados, nas partições, na semente, no espaço de busca ou nos limites do orçamento como resposta às métricas de validação. As intervenções relatadas a seguir são **correções de falhas de execução**, com impactos potenciais sobre tempo, comportamento numérico ou pré-processamento, explicitados em cada ocorrência. Embora algumas alterações tenham alcance global no código, **na recuperação da validação foram reexecutadas somente as avaliações que haviam falhado ou estavam ausentes**; os registros válidos das demais avaliações foram preservados na consolidação.

## 2. Resumo das ocorrências

| ID | Local / datasets afetados | Sintoma observado | Intervenção | Resultado verificado |
|---|---|---|---|---|
| F-01 | `363620` Bioresponse e `363677` hiva_agnostic | Falta de memória no TabICL; encerramento 137 em uma execução do hiva | Configuração de memória na recuperação e posterior implementação adaptativa em `src/models/group_model.py` | Recuperadas 18 avaliações do Bioresponse e 36 do hiva |
| F-02 | `363706` Taiwanese Bankruptcy, `group_tuned` | `ValueError: Input contains NaN` durante cálculo de AUC de uma tentativa Optuna | Tratamento específico de probabilidades não finitas em `src/pipeline/tune.py` | Recuperadas as 9 avaliações pendentes |
| F-03 | `363691` Online Shoppers, três baselines | `ValueError: Cannot cast object dtype to float64` envolvendo categoria como `Aug` | Codificação categórica aprendida no treino em `src/models/wrappers.py` | Recuperadas as 27 avaliações pendentes |
| F-04 | Sessões de computação e consolidação | Limites de uso de GPU, reinícios e resultados em pastas/contas distintas | Persistência em Drive, execuções por blocos e consolidação por chave | 1.350/1.350 avaliações válidas, sem duplicatas ou pendências nos CSVs finais |

## 3. Descrição das falhas e correções

### F-01 — Limite de memória em datasets de alta dimensionalidade

**Sintoma.** No Bioresponse (`363620`), os baselines haviam produzido **27/45** avaliações válidas, mas `group_default` e `group_tuned` falharam por OOM, deixando **18** avaliações pendentes. No hiva_agnostic (`363677`), havia **9/45** avaliações válidas; uma execução foi interrompida com código **137**, sugestivo de encerramento por falta de memória do processo/ambiente, mas sem comprovação de qual recurso específico esgotou. Permaneceram **36** avaliações pendentes, entre TabICL e baselines.

**Correção utilizada na recuperação original.** Em `src/models/group_model.py`, o `TabICLClassifier` passou inicialmente a ser construído com a seguinte configuração operacional, aplicada globalmente pelo construtor naquela versão:

```python
model_params = {
    "random_state": seed,
    "batch_size": 1,
    "disk_offload_dir": str(REPO_ROOT / "cache" / "tabicl_offload"),
    "inference_config": {
        "COL_CONFIG": {
            "max_pinned_memory_mb": 0.0,
            "use_async": False,
        },
    },
    **params,
}
```

Foi acrescentado `from data.load_tabarena import REPO_ROOT`. O mapeamento de `norm_methods` e o `search_space()` foram preservados.

O arquivo passou a utilizar a classe `AdaptiveTabICLClassifier`, com `HIGH_DIM_THRESHOLD = 500`. Quando `n_features_in_ > 500`, a classe ajusta `max_pinned_memory_mb = 0.0`, `use_async = False` e `disk_offload_dir` na configuração de inferência, além de usar temporariamente `batch_size = 1` em `_batch_forward`. Nos demais casos, preserva o comportamento padrão herdado do TabICL para esses ajustes operacionais.

**Verificação.** O primeiro teste no Bioresponse, `group_default`, fold externo 0, concluiu as três avaliações internas (`AUC interna = 0,8393` no log). Depois, os demais folds e o `group_tuned` do Bioresponse terminaram com cobertura **18/18** das avaliações anteriormente ausentes. No hiva, a retomada do TabICL e dos baselines concluiu **36/36** avaliações pendentes.

**Impacto metodológico.** A configuração utilizada na recuperação reduz a pressão de memória, mas pode aumentar `predict_time_s` e o tempo de ajuste/busca; a presença de `disk_offload_dir` não comprova, por si só, acesso efetivo ao disco. Como o Optuna tem limite de tempo, mudanças na duração das tentativas podem modificar quantas são concluídas e quais parâmetros são selecionados. A versão atual restringe os ajustes operacionais a bases com mais de 500 atributos. As avaliações já concluídas foram preservadas.

### F-02 — Probabilidades não finitas durante a busca Optuna

**Sintoma.** No Taiwanese Bankruptcy (`363706`), as versões padrão e os baselines haviam sido concluídos, mas o `group_tuned` permanecia com **36/45** avaliações válidas no dataset; faltavam **9** avaliações. O traceback apontou `roc_auc_score(...)` recebendo `NaN` em `proba[:, 1]`, dentro de `src/pipeline/tune.py`. A primeira tentativa com valor inválido encerrava a otimização.

**Correção implementada.** Foi criada a exceção `NonFiniteProbabilityError(ValueError)`. Em `_auc()`, é verificado `np.isfinite(proba).all()` antes da métrica. `study.optimize(...)` recebeu `catch=(NonFiniteProbabilityError,)`, permitindo que **somente esse tipo de tentativa inválida** seja registrado como `FAIL` e a busca continue. Ao término, o código exige pelo menos uma tentativa `COMPLETE`; caso contrário, sinaliza erro em vez de gerar um resultado artificial.

**Verificação.** Nos logs do Taiwanese, várias tentativas do Optuna foram marcadas como `FAIL` e outras concluíram. Os folds externos **0, 1 e 2** passaram, recuperando **9/9** avaliações internas com `status=ok`.

**Impacto metodológico.** Não foram alterados o espaço de busca nem o teto de 20 tentativas/600 segundos. O tratamento pode, entretanto, alterar o percurso da otimização e o número de tentativas válidas. **Embora a correção tenha alcance global no procedimento de busca**, a recuperação com esse tratamento foi restrita às **9 avaliações pendentes do `group_tuned` no Taiwanese Bankruptcy**; as avaliações válidas anteriores foram preservadas. A causa numérica específica das probabilidades não finitas não foi isolada: a intervenção corrigiu a forma de tratar tentativas inválidas, não a origem do `NaN` na biblioteca.

### F-03 — Incompatibilidade de atributos categóricos nos baselines

**Sintoma.** No Online Shoppers (`363691`), os dois sistemas TabICL estavam completos, mas LightGBM, XGBoost e CatBoost falharam nos três folds externos: **18/45** avaliações válidas, com **27** pendentes. O traceback de LightGBM mostrou a categoria textual `Aug` e terminou em `ValueError: Cannot cast object dtype to float64` durante validação de entrada do `pytabkit`, chamada pelo `ImputedClassifier`.

**Correção implementada.** Em `src/models/wrappers.py`, `ImputedClassifier` passou a:

1\. Identificar colunas categóricas e construir, em `fit()`, um mapeamento ordinal com **somente as categorias da partição de treino**.

2\. Reaplicar o mesmo mapeamento em `_transform()` durante treino e predição; categorias desconhecidas e valores ausentes recebem `-1`.

3\. Informar `cat_col_names` ao `fit()` quando esse argumento for suportado pelo estimador, para manter a identificação das colunas codificadas como categóricas.

4\. Preservar a imputação das colunas numéricas pela mediana do treino e os indicadores de ausência previamente definidos.

**Verificação.** Após a correção, o teste seletivo do LightGBM (fold externo 0) e a execução dos três baselines nos folds externos 0, 1 e 2 produziram as **27/27** avaliações que faltavam. Um erro anterior ao reparo foi mantido no arquivo de auditoria, mas não consta nos CSVs finais válidos.

**Impacto metodológico.** Trata-se de uma alteração real do pré-processamento, com **alcance global no código** para futuras execuções que utilizem `ImputedClassifier`. **Na recuperação, porém, foram executadas novamente apenas as 27 avaliações pendentes dos baselines no Online Shoppers**; as avaliações válidas dos demais datasets não foram refeitas. Como os mapeamentos categóricos são aprendidos apenas no `fit()` de cada partição de treino, não há uso das categorias da validação para construir o mapeamento.

### F-04 — Interrupções de ambiente e consolidação de resultados

**Sintoma.** Durante execuções com Python 3 e GPU NVIDIA T4 no Google Colab, houve limites de uso de GPU, reinícios de sessão e resultados produzidos em diferentes sessões/pastas de Drive.

**Medidas adotadas.** Os resultados foram gravados em pastas persistentes do Drive; as execuções de recuperação usaram `--task-ids`, `--models`, `--folds`, `--output-dir` e `--resume` para limitar a execução ao necessário. Os CSVs foram consolidados por chave `(task_id, fold, inner_fold, model)`, priorizando registros bem-sucedidos e preservando as falhas separadamente.

**Verificação.** O retorno `returncode=0` do comando não foi considerado suficiente isoladamente, porque o pipeline pode registrar falhas por unidade sem retornar código diferente de zero. O sucesso foi confirmado pela contagem das chaves únicas e pelo campo `status=ok`, além da correspondência entre os arquivos de validação e treino. A consolidação final comprovou **1.350/1.350 avaliações válidas**, com **45 por dataset**, **sem duplicatas e sem pendências**. As diferentes sessões usaram o ambiente de execução descrito no cabeçalho; a auditoria dos CSVs confirma a cobertura e a consistência dos resultados consolidados.

## 4. Auditoria da consolidação

Os dois arquivos finais são:

\- `results/validation.csv` — **1.350** registros válidos de validação interna.

\- `results/validation_train.csv` — **1.350** registros válidos correspondentes de treino.

Conferências executadas nos arquivos consolidados: unicidade da chave `(task_id, fold, inner_fold, model)`; coincidência de chaves entre treino e validação; **45 avaliações por dataset** (9 por sistema); ausência de unidades pendentes; campos de métricas e tempos válidos; `status=ok` para todas as linhas; coluna `error` vazia para os registros finais.&#x20;

**A validação interna está completa quanto à cobertura e à consistência dos dois CSVs consolidados.** O congelamento do protocolo e o teste final são etapas distintas deste registro.

## 5. Impactos sobre validade e comparabilidade

| Dimensão | Consideração para o relatório |
|---|---|
| Desempenho preditivo | Os reparos não foram motivados por AUC/accuracy; ainda assim, a codificação categórica e o comportamento das tentativas Optuna podem afetar as métricas. |
| Tempo e orçamento | Para datasets com mais de 500 atributos, `batch_size=1`, descarregamento e opções de transferência podem alterar os tempos. O limite de 600 s é verificado entre tentativas; `tune_time_s` efetivo pode ultrapassá-lo. |
| Heterogeneidade | Os CSVs agregam resultados obtidos em diferentes sessões do Google Colab com GPU T4. As correções de pré-processamento e tratamento de falhas podem afetar as métricas e os tempos de execução; o mesmo modelo de GPU também não garante tempos idênticos entre sessões. |
| Vazamento de dados | Os mapeamentos categóricos foram aprendidos dentro do `fit()` em cada partição de treino, sem utilizar as categorias da respectiva partição de validação. O protocolo mantém o teste externo separado da busca e da validação interna. |
| Reprodutibilidade | Os commits das correções e dos CSVs estão referenciados neste documento; `uv.lock`, diffs, logs, checkpoints e artefatos de predição complementam a rastreabilidade do experimento. |

## 6. Referências internas para auditoria

\- `docs/declaracoes_a_priori.md` — decisões de protocolo e limites para correções pós-declaração.

\- `src/models/group_model.py` — parâmetros operacionais de memória do TabICL.

\- `src/pipeline/tune.py` — validação das probabilidades e tratamento de tentativas inválidas do Optuna.

\- `src/models/wrappers.py` — codificação categórica e imputação dos baselines.

\- `src/pipeline/run_validation.py` — execução da validação interna e `--resume`.

\- `results/validation.csv` e `results/validation_train.csv` — resultados consolidados.

**Estado final deste registro:** validação interna consolidada, com **1.350/1.350 avaliações válidas**, **30/30 datasets completos** e **nenhuma avaliação pendente**. Correções e resultados registrados nos commits indicados no cabeçalho.
