# Diretrizes e Restrições de Protocolo - Projeto Final IN1102 (2026.2)

Este repositório implementa o protocolo experimental rigoroso da disciplina IN1102 (CIn/UFPE). As seguintes regras são **invioláveis**:

## 1. Integridade do Protocolo e Proibição de Vazamento de Dados (Data Leakage)
- **Teste Selado (Holdout Único):** O conjunto de teste de cada fold externo oficial do OpenML só pode ser avaliado **uma única vez** por `src.pipeline.final_test`.
- **Proibição de Consulta ao Teste:** NUNCA inspecione dados de teste, predições de teste (`results/preds/`) ou métricas de teste (`results/test.csv`) para guiar escolhas de hiperparâmetros, wrappers, espaços de busca, subconjuntos de datasets ou regras de pré-processamento.
- **Proibição de Uso do Leaderboard/OpenML Runs:** É expressamente proibido usar dados por dataset do leaderboard do TabArena ou runs/avaliações publicados no OpenML para selecionar datasets ou definir faixas do espaço de busca.
- **Validação é Apenas Diagnóstica:** Os resultados de `results/validation.csv` servem para diagnóstico e estudo de overfitting. Após executar `run_validation`, só são permitidas correções de erros de execução (`status = error` ou bugs de contrato), devidamente justificadas no relatório. Nenhuma mudança em modelos, wrappers ou espaço de busca pode ser motivada por desempenho na validação.
- **Pré-processamento Exclusivo do Treino:** Imputação, codificação, normalização e qualquer transformação supervisionada ou não supervisionada devem ser ajustadas estritamente dentro do método `fit(X, y)` de cada estimador/wrapper, usando unicamente os dados de treino da partição interna corrente.

## 2. Disciplina de Git e Auditoria (`check_protocol`)
- **Evidência no Histórico Git:** As garantias do protocolo dependem da cronologia de commits verificada por `python -m src.pipeline.check_protocol`.
- **Ordem Obrigatória de Commits:**
  1. Commit de todo código e seleção (`src/`, `data/selecao_datasets.txt`, `docs/declaracoes_a_priori.md`).
  2. Execução de `freeze` e commit de `results/protocolo.lock`.
  3. Envio para remoto (`git push`).
  4. Execução única de `final_test`.
  5. Commit imediato de `results/` (`test.csv`, `preds/`, `test_log.csv`, `test_manifest.json`) e `git push`.
- **Proibição de Reescrever Histórico:** NUNCA execute `git commit --amend`, `git rebase` ou `git push --force` após o commit do lock.
- **Modificações Pós-Freeze/Teste:** Se um arquivo travado precisar ser alterado após o teste (ex.: correção de bug de plot ou docstring), a mensagem de commit DEVE conter obrigatoriamente uma linha iniciando com `Justificativa: <motivo>`.
- **Execuções Adicionais de Teste:** O comando `final_test` só pode ser reexecutado com `--resume` para concluir combinações que falharam com erro ou foram interrompidas sob o mesmo lock, sem nunca recalcular linhas já concluídas.

## 3. Preservação de Assinaturas e Contratos de Código
- **Assinaturas Imutáveis:** NUNCA altere os nomes, parâmetros, valores padrão ou tipos de retorno de funções e classes marcadas como trabalho do grupo em `src/` e `tests/aceitacao/`. Os testes ocultos de correção dependem estritamente das assinaturas originais.
- **Contrato scikit-learn:** Estimadores em `group_model.py` e wrappers em `wrappers.py` devem cumprir estritamente o contrato: suportar rótulos textuais, atributos categóricos, categorias não vistas no treino, valores ausentes, predição de única instância (`predict_proba` retornando array 2D somando 1), `classes_`, `get_params` e `set_params` compatíveis com `sklearn.base.clone`.
- **Sem Dependências Prontas de Demšar:** A implementação em `demsar.py` deve ser autoral utilizando `numpy`, `pandas` e `scipy`. É proibido usar bibliotecas prontas que entregam o protocolo de Demšar (ex.: `autorank`, `scikit-posthocs`).

## 4. Ambiente de Execução e Testes
- **Gerenciador de Pacotes:** Utilize sempre `uv` (`uv run ...`, `uv sync ...`).
- **Modo Estrito de Testes:** Para verificar a conformidade dos testes de aceitação sem tolerância a `xfail`, execute com `$env:AM_STRICT="1"; uv run pytest tests/aceitacao` (PowerShell) ou `AM_STRICT=1 uv run pytest tests/aceitacao` (Bash).
- **Compatibilidade PowerShell:** Evite encadear comandos com `&&` no PowerShell do Windows; execute-os como comandos separados.

