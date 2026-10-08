"""Gráficos para o relatório: distribuição por sistema e custo versus desempenho.

O diagrama de diferença crítica está em `src.pipeline.demsar.plot_cd_diagram`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.pipeline.evaluate import HIGHER_IS_BETTER
from src.pipeline.results import ok_results


def boxplot_by_model(
    table: pd.DataFrame, metric: str = "auc_ovo", output: Path | None = None
) -> plt.Figure:
    """Boxplot da métrica por sistema (um ponto por dataset), ordenado pela mediana."""
    raise NotImplementedError("TODO(grupo): converta a tabela para formato longo e desenhe boxplot com os pontos sobrepostos")


def cost_vs_performance(
    raw: pd.DataFrame,
    metric: str = "auc_ovo",
    time_col: str = "total_time_s",
    output: Path | None = None,
) -> plt.Figure:
    """Tempo mediano por fold (escala log) versus métrica média, um ponto por sistema.

    Usa a mediana do tempo entre datasets, pois os tempos variam em ordens de
    grandeza conforme o tamanho do dataset.
    """
    raise NotImplementedError("TODO(grupo): agregue por sistema (média da métrica, mediana do tempo) e desenhe a dispersão em escala log")
