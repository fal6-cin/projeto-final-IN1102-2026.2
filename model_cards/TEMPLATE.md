# Model Card: <Nome do Modelo>

> Copie este arquivo para `model_cards/<nome-do-modelo>.md` e preencha-o para o modelo tema atribuído ao grupo (IN1102, pós-graduação, 2026.2). Substitua os campos `<...>` pelos valores reais. Não deixe campos em branco; use "N/A" quando não se aplicar e explique o motivo.
>
> Estrutura inspirada em Mitchell et al. (2019), com extensões da disciplina: fatores observados nos regimes do TabArena-v0.1, protocolo de treino, validação e teste, análise quantitativa segundo Demšar (2006) contra os baselines LightGBM, XGBoost e CatBoost, estudo de overfitting e viés-variância e seção de avisos e recomendações. Os números de desempenho final vêm de `results/test.csv` (avaliação final única) e das saídas de `notebooks/04_analise_estatistica.ipynb`; os números de desenvolvimento e de overfitting vêm de `results/validation.csv`, `results/validation_train.csv` e de `notebooks/05_overfitting.ipynb`. Indique sempre de qual arquivo vem cada número.

## 1. Detalhes do modelo

- **Nome e versão:** <ex.: TabICL v2, pacote tabicl 2.x>
- **Autores e artigo original:** <autores, ano, veículo>
- **Repositório oficial:** <URL>
- **Licença do código:** <ex.: MIT, Apache 2.0, BSD>
- **Licença dos pesos pré-treinados (se aplicável):** <ex.: não comercial, com aceite de licença>
- **Família arquitetural:** <ex.: modelo de fundação com aprendizado em contexto; MLP; aditivo generalizado; vizinhança aprendida>
- **Contagem de parâmetros:** <total; treináveis vs. fixos, quando aplicável>
- **Complexidade computacional:** <tempo e memória em função de n (instâncias) e p (atributos), em treino e em inferência>
- **Pico de memória observado:** <RAM ou VRAM, e em qual dataset>
- **Extra e dependências:** <extra do pyproject.toml e versões do uv.lock; CUDA, se usado>
- **Ponto de entrada no repositório:** <classe usada em `build_group_model`; wrapper, se houver>
- **Configuração padrão (`group_default`):** <hiperparâmetros e a fonte da recomendação>
- **Espaço de busca (`group_tuned`):** <hiperparâmetros, tipos, faixas, escalas e justificativa a priori (artigo, documentação, espaços publicados pelo TabArena); origem de cada faixa. Os resultados por dataset do leaderboard do TabArena não podem ser usados>

## 2. Uso pretendido

- **Caso de uso primário:** classificação supervisionada em dados tabulares.
- **Casos de uso fora de escopo:** <ex.: dados não IID, séries temporais, imagens, texto bruto>
- **Usuários pretendidos:** <ex.: pesquisadores e profissionais em problemas tabulares>
- **Faixa de n suportada:** <limites práticos observados e recomendados pelos autores>
- **Faixa de p e de classes suportada:** <limites de atributos e de classes; tratamento de categóricos>
- **Condições operacionais:** <CPU vs. GPU, memória mínima, tempo típico por fold>

## 3. Fatores observados

Dimensões em que o desempenho varia, avaliadas nos 30 datasets selecionados pelo grupo (análise descritiva):

- **Tamanho do dataset:** <pequeno (n < 2.500), médio (2.500 ≤ n < 10.000), grande (n ≥ 10.000)>
- **Número de classes:** <binário vs. multiclasse>
- **Proporção de atributos categóricos:** <baixa (< 25%) vs. alta (≥ 25%)>
- **Valores ausentes:** <com vs. sem; tratamento adotado (nativo, `ImputedClassifier` ou outro)>
- **Outros fatores relevantes:** <ex.: desbalanceamento, alta dimensionalidade>

## 4. Métricas alcançadas (avaliação final, `results/test.csv`)

Uma medida por dataset (média dos testes dos 3 folds externos); depois, média e desvio entre os N datasets completos. O posto médio é calculado entre os 4 sistemas do teste principal (`group_tuned` e os três baselines).

Os intervalos de 95% por bootstrap percentil (1.000 reamostras das predições de teste) são calculados por dataset e fold e apresentados no apêndice do relatório. A média dos limites desses intervalos entre datasets **não** é um intervalo de confiança da média entre datasets; por isso, a tabela resume a incerteza pela mediana da largura dos intervalos.

| Métrica | `group_default` | `group_tuned` | Mediana da largura do IC 95% (bootstrap, `group_tuned`) | Posto médio (`group_tuned`) |
|---|---|---|---|---|
| AUC OvO | <0,0000 ± 0,0000> | <0,0000 ± 0,0000> | <0,0000> | <0,00> |
| Acurácia (ACC) | <0,0000 ± 0,0000> | <0,0000 ± 0,0000> | <0,0000> | <0,00> |
| G-Mean | <0,0000 ± 0,0000> | <0,0000 ± 0,0000> | <0,0000> | <0,00> |
| Entropia cruzada (CE) | <0,0000 ± 0,0000> | <0,0000 ± 0,0000> | <0,0000> | <0,00> |
| Tempo total por fold (s, mediana) | <0,0> | <0,0> | N/A | N/A |

Decomposição do custo de `group_tuned` (mediana por fold): busca <0,0 s>, treino <0,0 s>, predição <0,0 s>. O tempo de busca é o tempo efetivo (`tune_time_s`), que pode exceder o limite de 600 s porque o limite é verificado entre tentativas; informe o máximo observado: <0,0 s>.

**Validação interna vs. teste (AUC OvO, média entre datasets):** `group_default` <validação> vs. <teste>; `group_tuned` <validação, otimista> vs. <teste>. <A diferença indica otimismo da fase de desenvolvimento?>

### Resultados por regime (AUC OvO no teste, média entre datasets; n de datasets entre parênteses)

- **Tamanho:** pequeno: <...> (<n>); médio: <...> (<n>); grande: <...> (<n>)
- **Número de classes:** binário: <...> (<n>); multiclasse: <...> (<n>)
- **Proporção categórica:** baixa: <...> (<n>); alta: <...> (<n>)
- **Valores ausentes:** com: <...> (<n>); sem: <...> (<n>)

## 5. Dados de avaliação e protocolo

- **Origem:** 30 datasets de classificação do TabArena-v0.1 (suíte OpenML 457), escolhidos pelo grupo a partir do catálogo, antes de qualquer resultado.
- **Distribuição por regime:** 10 pequenos + 10 médios + 10 grandes; <binários e multiclasse por regime>.
- **Critérios da seleção:** <resumo da justificativa; detalhes no relatório>
- **Divisão dos dados (CV aninhada):** CV externa com as partições oficiais das tasks do OpenML (repetição 0, 3 folds estratificados); CV interna de 3 folds estratificados no treino de cada fold externo (subamostra de até 10.000), usada pela busca e por todos os estudos.
- **Papel de `validation.csv`:** apenas diagnóstico (o treino de um fold externo contém o teste dos outros); correções feitas depois de `run_validation`, todas de falha de execução: <nenhuma, ou quais, com causa e alteração>.
- **Congelamento:** `results/protocolo.lock` criado em <data> no commit <hash>, com o sha256 de todos os arquivos versionados em `src/` e `data/` (`src/**/*.py`, `data/*.py` e `data/selecao_datasets.txt`); sha256 do lock: <...>; commit do lock: <hash>; recongelamentos antes do teste: <nenhum, ou quais e por quê>.
- **Teste usado uma única vez:** execuções registradas em `results/test_log.csv`: <avaliação inicial e retomadas (`resume=True`)>; justificativa de cada retomada: <...>; commit dos resultados: <hash>; `check_protocol`: <APROVADO, avisos>.
- **Decisões declaradas a priori:** <espaço de busca, subconjunto de datasets dos estudos, grades> (registradas em `docs/declaracoes_a_priori.md`, commit <hash>)
- **Datasets excluídos da análise estatística:** <quais e por quê (regra de completude)>
- **Pré-processamento:** <imputação, codificação de categóricos, escalonamento, subamostragem do contexto (se houver); todos ajustados apenas no treino, dentro de `fit`>
- **Lista dos datasets:** <ver tabela do relatório (nome, task ID, n, atributos, classes, regime)>

## 6. Dados de treino e pré-treino

- **O modelo é pré-treinado, treinado do zero ou híbrido?** <responder>
- **Dados de pré-treino (se aplicável):** <ex.: datasets sintéticos gerados por priors; dados reais>
- **Risco de sobreposição com os datasets de avaliação:** <há indício de que algum dataset do TabArena tenha sido usado no pré-treino ou na meta-aprendizagem? discutir também para os padrões TD dos baselines>
- **Vieses possivelmente herdados do pré-treino:** <descrever>

## 7. Análise quantitativa (Demšar, 2006)

Sobre `results/test.csv`. Métrica primária: AUC OvO, declarada antes da análise. N = <número de datasets completos>, k = 4, alfa = 0,05.

- **Friedman com Iman-Davenport:** chi2_F = <...>, F_F = <...> com (<k-1>, <(k-1)(N-1)>) graus de liberdade, p = <...>. Decisão (por F_F): <rejeita ou não H0>.
- **Controle `group_tuned`, Holm:** <para cada baseline: diferença de postos, z, p e p ajustado; conclusão. Se F_F não rejeitou H0, apresente os valores apenas como descrição, sem conclusão de diferença significativa>
- **Bonferroni-Dunn:** CD = <...>; <quais baselines ficam fora do intervalo do controle>; <coincide com Holm?> <mesma condição: interpretado como teste só se F_F rejeitou H0>
- **Nemenyi e diagrama CD (complementar):** CD = <...>; <grupos de sistemas sem diferença significativa> <mesma condição>
- **Padrão vs. com busca (Wilcoxon):** número de diferenças usado = <...> (de N = <...>), R+ = <...>, R- = <...>, T = <...>, p = <...>; vitórias/empates/derrotas = <.../.../...>; <o ganho compensa o tempo de busca?>
- **Robustez (Wilcoxon com Holm por par, `wilcoxon_vs_control`):** <resultado; a discussão das divergências em relação à análise baseada em postos médios é opcional>
- **Métrica secundária:** <repetição para CE ou outra métrica; mudanças nas conclusões; multiplicidade>
- **Regimes e custo:** <onde o modelo se destaca e onde perde; provável explicação ligada à arquitetura; posição na relação custo vs. desempenho>
- **Sensibilidade e estudo específico (modelos com API scikit-learn nativa, por CV interna):** <2 hiperparâmetros estudados, grade, datasets usados e leitura; aspecto do modelo estudado e resultado. Grupos de TabDPT-1.3, ModernNCA e TabR: N/A>
- **Diferença de magnitude vs. significância:** <diferenças estatisticamente significativas são relevantes na prática?>

## 8. Overfitting e viés-variância

Fase de desenvolvimento (CV interna no treino dos folds externos; o teste não é usado).

- **Lacuna de generalização (`generalization_gap`, AUC OvO, treino menos validação):** `group_default` <gap_mean ± gap_std>; `group_tuned` <...> (validação otimista); LightGBM <...>; XGBoost <...>; CatBoost <...>. <Por regime de tamanho: onde a lacuna é maior?>
- **Métrica de treino:** medida numa subamostra de até 2.000 instâncias do treino interno. <Se o modelo é de contexto ou de vizinhança, explique por que a métrica de treino é quase trivial e como isso afeta a leitura da lacuna.>
- **Curva de complexidade do modelo tema:** hiperparâmetro <...>, grade <...>, sentido de simplicidade <...>, datasets <um por regime>; melhor média = <...>; regra de 1 EP = <...>. <O valor escolhido está dentro do espaço de busca declarado?>
- **Curva de `num_leaves` do LightGBM:** melhor média = <...>; regra de 1 EP = <...> por dataset.
- **Erros por classe (validação interna, `results/preds_val/`):** <classes com menor recall por sistema nos datasets do estudo; relação com o G-Mean; diferenças entre o modelo tema e os baselines>.
- **Leitura viés-variância:** <em que região da curva fica a configuração padrão? o modelo tende a subajuste ou a sobreajuste nos regimes estudados?> Os estudos são diagnósticos e não alteraram o espaço de busca nem o lock.

## 9. Considerações éticas

- **Riscos de uso indevido:** <ex.: decisões opacas em domínios sensíveis>
- **Desempenho por classe:** <recall por classe; classes minoritárias com baixo recall; relação com o G-Mean>
- **Dependência de licença de pesos:** <restrições de uso comercial ou de redistribuição>
- **Impacto ambiental:** <tempo total de computação nas duas fases, uso de GPU, custo da busca>
- **Recomendações de auditoria:** <ex.: comparar com um modelo interpretável (EBM, APLR) antes de uso em produção>

## 10. Avisos e recomendações

- **Quando usar este modelo:** <regimes com melhor desempenho ou melhor relação custo vs. desempenho>
- **Quando não usar este modelo:** <regimes em que os baselines vencem; restrições operacionais>
- **Alternativas recomendadas:** <por regime ou restrição>
- **Ameaças à validade:** <assimetria de busca (baselines sem busca), imputação, subamostragem, falhas de execução, poucos datasets por célula de regime, sobreposição com pré-treino, otimismo da validação interna de `group_tuned`, estudos em um único fold externo>

## 11. Reprodutibilidade

- **Ambiente:** Python 3.12, `uv.lock` do repositório; extras: <baselines + ...>
- **Hardware:** <CPU, GPU, RAM; partes executadas no Colab, se houver>
- **Tempo total de execução:** <horas no desenvolvimento e na avaliação final>
- **Comandos:**
  ```bash
  uv sync --frozen --extra baselines --extra <modelo>
  uv run python -m src.pipeline.check_selection
  uv run python -m src.pipeline.run_validation
  git add -A && git commit -m "Código final do desenvolvimento"
  uv run python -m src.pipeline.freeze
  git add results/protocolo.lock && git commit -m "Congela o protocolo"
  uv run python -m src.pipeline.final_test
  git add results && git commit -m "Avaliação final"
  uv run python -m src.pipeline.check_protocol
  AM_STRICT=1 uv run pytest
  ```
- **Hash do commit congelado:** <git_commit de results/protocolo.lock>
- **Arquivos do protocolo:** `results/protocolo.lock` (comitado antes dos resultados), `results/test_log.csv`, `results/test_manifest.json` versionados: <sim/não>; saída de `check_protocol`: <APROVADO/REPROVADO>
- **Situação dos testes:** <todos os testes de aceitação passando em modo estrito? CI verde?>

## 12. Referências

- <artigo original do modelo>
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. FAT*.
- Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR, 7, 1-30.
- Benavoli, A., Corani, G., Mangili, F. (2016). Should We Really Use Post-Hoc Tests Based on Mean-Ranks? JMLR, 17(5), 1-10.
- Cawley, G. C., Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR, 11, 2079-2107.
- Varma, S., Simon, R. (2006). Bias in Error Estimation When Using Cross-Validation for Model Selection. BMC Bioinformatics, 7, 91.
- Hastie, T., Tibshirani, R., Friedman, J. (2009). The Elements of Statistical Learning, 2. ed. Springer.
- Breiman, L., Friedman, J. H., Olshen, R. A., Stone, C. J. (1984). Classification and Regression Trees. Wadsworth.
- Holzmüller, D., Grinsztajn, L., Steinwart, I. (2024). Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data. NeurIPS.
- TabArena-v0.1 (NeurIPS 2025): https://tabarena.ai
