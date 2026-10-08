"""Comparação estatística de classificadores em múltiplos datasets (Demšar, 2006).

Referência: Demšar, J. (2006). Statistical Comparisons of Classifiers over
Multiple Data Sets. Journal of Machine Learning Research, 7, 1-30.

Entrada: tabela `dataset x sistema` com UMA medida por dataset (a média dos
folds externos na avaliação final, `results/test.csv`; Seção 3.2.3 do artigo).
Os folds de um mesmo dataset nunca entram como observações independentes. A
análise é feita uma única vez, após `final_test`, e não retroalimenta o modelo.

Procedimento da disciplina (IN1102, pós-graduação):
    1. postos por dataset (1 = melhor; empates recebem o posto médio);
    2. teste de Friedman com a correção de Iman-Davenport (decisão por F_F);
    3. modelo do grupo como controle: Holm sobre as estatísticas z dos postos
       médios e diagrama de Bonferroni-Dunn;
    4. análise complementar todos contra todos: Nemenyi e diagrama CD;
    5. comparação de dois sistemas (padrão vs. com busca de hiperparâmetros): Wilcoxon
       signed-ranks (descritiva do resultado final; não retroalimenta a escolha do modelo).

Todas as funções recebem `higher_is_better`: verdadeiro para AUC, acurácia e
G-Mean; falso para entropia cruzada e tempos.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FriedmanResult:
    """Resultado do teste de Friedman (Demšar, 2006, Seção 3.2.2)."""

    avg_ranks: pd.Series  # posto médio de cada sistema (menor = melhor)
    n_datasets: int  # N
    k: int  # número de sistemas
    chi2_f: float  # estatística de Friedman
    p_value_chi2: float  # p-valor de chi2_F com k - 1 graus de liberdade
    f_f: float  # estatística de Iman-Davenport
    df1: int  # k - 1
    df2: int  # (k - 1)(N - 1)
    p_value_ff: float  # p-valor de F_F (critério de decisão)


def rank_matrix(table: pd.DataFrame, higher_is_better: bool = True) -> pd.DataFrame:
    """Postos dos sistemas em cada dataset (linha): 1 = melhor, empates com posto médio."""
    raise NotImplementedError("TODO(grupo): ranqueie cada linha (dataset) com postos médios para empates; use scipy.stats.rankdata")


def average_ranks(table: pd.DataFrame, higher_is_better: bool = True) -> pd.Series:
    """Posto médio R_j de cada sistema sobre os N datasets."""
    raise NotImplementedError("TODO(grupo): média, sobre os datasets, da matriz de postos")


def friedman_from_ranks(ranks: pd.DataFrame) -> FriedmanResult:
    """Friedman e Iman-Davenport a partir da matriz de postos N x k.

    Usa as fórmulas do artigo, sem correção de empates. Quando chi2_F atinge o
    máximo N(k - 1) (concordância total entre os datasets), o denominador de F_F
    se anula: devolva F_F = inf e p_value_ff = 0.
    """
    raise NotImplementedError("TODO(grupo): implemente chi2_F, F_F e seus p-valores (Demšar, 2006, Seção 3.2.2)")


def friedman_test(table: pd.DataFrame, higher_is_better: bool = True) -> FriedmanResult:
    """Teste de Friedman com correção de Iman-Davenport sobre a tabela dataset x sistema."""
    return friedman_from_ranks(rank_matrix(table, higher_is_better))


def q_alpha(k: int, alpha: float = 0.05, method: str = "nemenyi") -> float:
    """Valor crítico q_alpha (Demšar, 2006, Tabela 5).

    method='nemenyi': amplitude studentizada (infinitos graus de liberdade) dividida por sqrt(2).
    method='bonferroni-dunn': quantil normal com alpha dividido por k - 1 comparações.
    """
    raise NotImplementedError("TODO(grupo): use scipy.stats.studentized_range (Nemenyi) e scipy.stats.norm (Bonferroni-Dunn)")


def critical_difference(
    k: int, n_datasets: int, alpha: float = 0.05, method: str = "nemenyi"
) -> float:
    """Diferença crítica CD = q_alpha * sqrt(k(k+1) / (6N))."""
    raise NotImplementedError("TODO(grupo): aplique a fórmula da CD (Demšar, 2006, Seção 3.2.2)")


def nemenyi_pairs(avg_ranks: pd.Series, n_datasets: int, alpha: float = 0.05) -> pd.DataFrame:
    """Todos os pares de sistemas: diferença de postos médios, CD e decisão (Nemenyi).

    Uma linha por par não ordenado, k(k - 1)/2 linhas, com as colunas model_a,
    model_b, rank_diff (|R_a - R_b|), cd e significant (rank_diff >= cd).
    """
    raise NotImplementedError("TODO(grupo): para cada par (a, b), |R_a - R_b| comparado com a CD de Nemenyi")


def holm(pvalues) -> np.ndarray:
    """p-valores ajustados pelo procedimento de Holm (step-down)."""
    raise NotImplementedError("TODO(grupo): ordene os p-valores; o i-ésimo menor (i = 1, ..., m) é multiplicado por (m - i + 1); imponha monotonicidade e limite 1")


def holm_vs_control(
    avg_ranks: pd.Series, n_datasets: int, control: str, alpha: float = 0.05
) -> pd.DataFrame:
    """Controle contra cada outro sistema: z dos postos médios, p bilateral e ajuste de Holm.

    z = (R_i - R_controle) / sqrt(k(k + 1) / (6N)). Uma linha por sistema diferente
    do controle, com as colunas control, model, rank_diff, z, p_value, p_holm e
    significant (p_holm < alpha).
    """
    raise NotImplementedError("TODO(grupo): z = (R_i - R_controle) / sqrt(k(k+1)/(6N)), p = 2 * norm.sf(|z|), depois Holm")


def wilcoxon_signed_ranks(a, b, higher_is_better: bool = True) -> dict[str, float]:
    """Wilcoxon signed-ranks para dois sistemas em N datasets (Demšar, 2006, Seção 3.1.3).

    d_i = a_i - b_i (sinal invertido se menor é melhor). Os postos das diferenças
    nulas são divididos igualmente entre R+ e R-; se o número de zeros for ímpar,
    um deles é descartado antes de tudo (inclusive do p-valor). T = min(R+, R-).
    O p-valor bilateral é o de `scipy.stats.wilcoxon(d, zero_method='zsplit')`
    sobre as diferenças restantes (p = 1 se todas forem nulas).

    Returns:
        dict com as chaves n (diferenças usadas), r_plus, r_minus, T, p_value,
        wins, ties e losses (contagens de d > 0, d = 0 e d < 0).
    """
    raise NotImplementedError("TODO(grupo): calcule R+, R-, T e o p-valor; conte vitórias, empates e derrotas de a")


def wilcoxon_vs_control(
    table: pd.DataFrame, control: str, higher_is_better: bool = True, alpha: float = 0.05
) -> pd.DataFrame:
    """Wilcoxon do controle contra cada outro sistema, com ajuste de Holm (análise de robustez).

    Uma linha por sistema, com control, model, as chaves de `wilcoxon_signed_ranks`,
    p_holm e significant (p_holm < alpha).
    """
    raise NotImplementedError("TODO(grupo): aplique wilcoxon_signed_ranks para cada sistema e ajuste os p-valores por Holm")


def cd_cliques(avg_ranks: pd.Series, cd: float) -> list[list[str]]:
    """Grupos maximais de sistemas consecutivos (por posto) cuja amplitude de postos é menor que a CD."""
    raise NotImplementedError("TODO(grupo): ordene por posto e encontre os intervalos maximais com diferença < CD")


def plot_cd_diagram(
    avg_ranks: pd.Series,
    cd: float,
    control: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Diagrama de diferença crítica (Demšar, 2006, Figura 1, painéis a e b).

    Eixo de postos invertido (melhores à direita). Sem `control`, barras ligam os
    grupos sem diferença significativa (Nemenyi). Com `control`, marca-se o
    intervalo de +- CD em torno do posto do controle (Bonferroni-Dunn).
    """
    raise NotImplementedError("TODO(grupo): desenhe o eixo de postos, os rótulos, a barra da CD e os grupos (ou o intervalo do controle)")
