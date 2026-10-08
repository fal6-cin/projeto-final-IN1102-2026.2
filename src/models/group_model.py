"""Modelo tema do grupo: construção e espaço de busca de hiperparâmetros.

O grupo implementa duas funções:

- `build_group_model(seed, **params)`: devolve um estimador compatível com
  scikit-learn (`fit`, `predict_proba`, `classes_`, `get_params`). Sem `params`,
  usa a configuração padrão recomendada pelos autores (sistema `group_default`);
  com `params`, usa os hiperparâmetros encontrados na busca (`group_tuned`).
- `search_space(trial)`: devolve o dicionário de hiperparâmetros amostrados pelo
  Optuna. Faixas, escalas e fontes devem ser justificadas no relatório.

O estimador deve funcionar com rótulos de texto, atributos categóricos
(`pandas.Categorical`), valores ausentes, problemas multiclasse, categorias que
não apareceram no treino e predição de uma única instância. O teste de
contrato em `tests/aceitacao/test_modelo_grupo.py` verifica esses requisitos.

Ponto de partida para cada modelo tema (instale o extra indicado; ver README):

| Modelo          | Extra     | Ponto de entrada                                             |
|-----------------|-----------|--------------------------------------------------------------|
| TabPFN-3.5      | tabpfn    | tabpfn.TabPFNClassifier (ModelVersion.V3_5_FAST em CPU)      |
| TabICL v2       | tabicl    | tabicl.TabICLClassifier                                      |
| TabDPT-1.3      | tabdpt    | src.models.wrappers.TabDPTWrapper (esqueleto a completar)    |
| TabM            | baselines | pytabkit.TabM_D_Classifier                                   |
| RealMLP         | baselines | pytabkit.RealMLP_TD_Classifier                               |
| FT-Transformer  | baselines | pytabkit.FTT_D_Classifier                                    |
| xRFM            | baselines | pytabkit.XRFM_D_Classifier                                   |
| ModernNCA       | modernnca | src.models.wrappers.ModernNCAWrapper (esqueleto a completar) |
| EBM             | ebm       | interpret.glassbox.ExplainableBoostingClassifier             |
| APLR            | aplr      | aplr.APLRClassifier                                          |
| TabR            | baselines | src.models.wrappers.TabRWrapper (esqueleto a completar)      |

Os modelos do pytabkit não aceitam valores ausentes em colunas numéricas: use
`src.models.wrappers.ImputedClassifier`, como nos baselines.
"""

from __future__ import annotations

from typing import Any

import optuna


def build_group_model(seed: int = 42, **params: Any):
    """Constrói o modelo tema do grupo com a semente e os hiperparâmetros indicados."""
    raise NotImplementedError("TODO(grupo): implemente o modelo tema em src/models/group_model.py")


def search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Espaço de busca do modelo tema (hiperparâmetros aceitos por build_group_model)."""
    raise NotImplementedError("TODO(grupo): defina o espaço de busca em src/models/group_model.py")
