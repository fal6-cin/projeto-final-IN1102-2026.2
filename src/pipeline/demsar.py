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

import itertools
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
    sign = -1.0 if higher_is_better else 1.0
    ranked = np.apply_along_axis(
        lambda r: stats.rankdata(sign * r, method="average"),
        axis=1,
        arr=table.values,
    )
    return pd.DataFrame(ranked, index=table.index, columns=table.columns, dtype=float)


def average_ranks(table: pd.DataFrame, higher_is_better: bool = True) -> pd.Series:
    """Posto médio R_j de cada sistema sobre os N datasets."""
    return rank_matrix(table, higher_is_better=higher_is_better).mean(axis=0)


def friedman_from_ranks(ranks: pd.DataFrame) -> FriedmanResult:
    """Friedman e Iman-Davenport a partir da matriz de postos N x k.

    Usa as fórmulas do artigo, sem correção de empates. Quando chi2_F atinge o
    máximo N(k - 1) (concordância total entre os datasets), o denominador de F_F
    se anula: devolva F_F = inf e p_value_ff = 0.
    """
    n_datasets, k = ranks.shape
    avg_r = ranks.mean(axis=0)
    sum_sq = float(np.sum(avg_r**2))
    chi2_f = float((12.0 * n_datasets / (k * (k + 1))) * (sum_sq - (k * (k + 1) ** 2) / 4.0))
    p_chi2 = float(stats.chi2.sf(chi2_f, k - 1))

    df1 = k - 1
    df2 = (k - 1) * (n_datasets - 1)
    denom = n_datasets * (k - 1) - chi2_f
    if denom <= 1e-12:
        f_f = float("inf")
        p_ff = 0.0
    else:
        f_f = float((n_datasets - 1) * chi2_f / denom)
        p_ff = float(stats.f.sf(f_f, df1, df2))

    return FriedmanResult(
        avg_ranks=avg_r,
        n_datasets=n_datasets,
        k=k,
        chi2_f=chi2_f,
        p_value_chi2=p_chi2,
        f_f=f_f,
        df1=df1,
        df2=df2,
        p_value_ff=p_ff,
    )


def friedman_test(table: pd.DataFrame, higher_is_better: bool = True) -> FriedmanResult:
    """Teste de Friedman com correção de Iman-Davenport sobre a tabela dataset x sistema."""
    return friedman_from_ranks(rank_matrix(table, higher_is_better))


def q_alpha(k: int, alpha: float = 0.05, method: str = "nemenyi") -> float:
    """Valor crítico q_alpha (Demšar, 2006, Tabela 5).

    method='nemenyi': amplitude studentizada (infinitos graus de liberdade) dividida por sqrt(2).
    method='bonferroni-dunn': quantil normal com alpha dividido por k - 1 comparações.
    """
    if method == "nemenyi":
        return float(stats.studentized_range.ppf(1.0 - alpha, k, df=np.inf) / np.sqrt(2.0))
    elif method == "bonferroni-dunn":
        return float(stats.norm.ppf(1.0 - alpha / (2.0 * (k - 1))))
    else:
        raise ValueError(f"Método desconhecido: {method}")


def critical_difference(
    k: int, n_datasets: int, alpha: float = 0.05, method: str = "nemenyi"
) -> float:
    """Diferença crítica CD = q_alpha * sqrt(k(k+1) / (6N))."""
    q = q_alpha(k, alpha=alpha, method=method)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_datasets)))


def nemenyi_pairs(avg_ranks: pd.Series, n_datasets: int, alpha: float = 0.05) -> pd.DataFrame:
    """Todos os pares de sistemas: diferença de postos médios, CD e decisão (Nemenyi).

    Uma linha por par não ordenado, k(k - 1)/2 linhas, com as colunas model_a,
    model_b, rank_diff (|R_a - R_b|), cd e significant (rank_diff >= cd).
    """
    k = len(avg_ranks)
    cd = critical_difference(k, n_datasets, alpha=alpha, method="nemenyi")
    rows = []
    for a, b in itertools.combinations(avg_ranks.index, 2):
        diff = abs(float(avg_ranks[a] - avg_ranks[b]))
        rows.append({
            "model_a": a,
            "model_b": b,
            "rank_diff": diff,
            "cd": cd,
            "significant": bool(diff >= cd),
        })
    return pd.DataFrame(rows)


def holm(pvalues) -> np.ndarray:
    """p-valores ajustados pelo procedimento de Holm (step-down)."""
    pvals = np.asarray(pvalues, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj = np.empty(m, dtype=float)
    for i in range(m):
        adj[i] = sorted_p[i] * (m - i)
    for i in range(1, m):
        adj[i] = max(adj[i], adj[i - 1])
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def holm_vs_control(
    avg_ranks: pd.Series, n_datasets: int, control: str, alpha: float = 0.05
) -> pd.DataFrame:
    """Controle contra cada outro sistema: z dos postos médios, p bilateral e ajuste de Holm.

    z = (R_i - R_controle) / sqrt(k(k + 1) / (6N)). Uma linha por sistema diferente
    do controle, com as colunas control, model, rank_diff, z, p_value, p_holm e
    significant (p_holm < alpha).
    """
    k = len(avg_ranks)
    se = np.sqrt(k * (k + 1) / (6.0 * n_datasets))
    rows = []
    other_models = [m for m in avg_ranks.index if m != control]
    for model in other_models:
        diff = float(avg_ranks[model] - avg_ranks[control])
        z = diff / se
        p_val = float(2.0 * stats.norm.sf(abs(z)))
        rows.append({
            "control": control,
            "model": model,
            "rank_diff": abs(diff),
            "z": z,
            "p_value": p_val,
        })
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_value"])
    df["significant"] = df["p_holm"] < alpha
    return df


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
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b if higher_is_better else b - a

    wins = int(np.sum(d > 0))
    ties = int(np.sum(d == 0))
    losses = int(np.sum(d < 0))

    if ties % 2 != 0:
        zero_idx = np.where(d == 0)[0][0]
        d = np.delete(d, zero_idx)

    n = len(d)
    if n == 0 or np.all(d == 0):
        return {
            "n": n,
            "r_plus": 0.0,
            "r_minus": 0.0,
            "T": 0.0,
            "p_value": 1.0,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        }

    ranks = stats.rankdata(np.abs(d), method="average")
    r_plus = float(np.sum(ranks[d > 0]) + 0.5 * np.sum(ranks[d == 0]))
    r_minus = float(np.sum(ranks[d < 0]) + 0.5 * np.sum(ranks[d == 0]))
    t_stat = min(r_plus, r_minus)
    res = stats.wilcoxon(d, zero_method="zsplit")
    p_val = float(res.pvalue)

    return {
        "n": n,
        "r_plus": r_plus,
        "r_minus": r_minus,
        "T": t_stat,
        "p_value": p_val,
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def wilcoxon_vs_control(
    table: pd.DataFrame, control: str, higher_is_better: bool = True, alpha: float = 0.05
) -> pd.DataFrame:
    """Wilcoxon do controle contra cada outro sistema, com ajuste de Holm (análise de robustez).

    Uma linha por sistema, com control, model, as chaves de `wilcoxon_signed_ranks`,
    p_holm e significant (p_holm < alpha).
    """
    rows = []
    other_models = [m for m in table.columns if m != control]
    for model in other_models:
        w = wilcoxon_signed_ranks(
            table[control], table[model], higher_is_better=higher_is_better
        )
        rows.append({"control": control, "model": model, **w})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_value"])
    df["significant"] = df["p_holm"] < alpha
    return df


def cd_cliques(avg_ranks: pd.Series, cd: float) -> list[list[str]]:
    """Grupos maximais de sistemas consecutivos (por posto) cuja amplitude de postos é menor que a CD."""
    sorted_avg = avg_ranks.sort_values()
    models = list(sorted_avg.index)
    ranks = list(sorted_avg.values)
    n = len(models)
    cliques = []
    for i in range(n):
        for j in range(i, n):
            if ranks[j] - ranks[i] < cd:
                can_extend_left = i > 0 and (ranks[j] - ranks[i - 1] < cd)
                can_extend_right = j < n - 1 and (ranks[j + 1] - ranks[i] < cd)
                if not can_extend_left and not can_extend_right:
                    clique = models[i : j + 1]
                    if len(clique) > 1 and clique not in cliques:
                        cliques.append(clique)
    return cliques


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
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    k = len(avg_ranks)
    ax.set_xlim(k + 0.5, 0.5)
    ax.set_xticks(range(1, k + 1))
    ax.hlines(0, 1, k, color="black", linewidth=1.5)
    ax.vlines(range(1, k + 1), -0.05, 0.05, color="black", linewidth=1.0)

    sorted_avg = avg_ranks.sort_values()
    for idx, (model, r) in enumerate(sorted_avg.items()):
        ax.plot(r, 0, "o", color="tab:blue", markersize=6)
        y_text = 0.2 if idx % 2 == 0 else -0.25
        ax.annotate(
            f"{model}\n({r:.3f})",
            xy=(r, 0),
            xytext=(r, y_text),
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.8},
        )

    if control is None:
        cliques = cd_cliques(avg_ranks, cd)
        for i, c in enumerate(cliques):
            c_ranks = [avg_ranks[m] for m in c]
            y_bar = -0.5 - 0.15 * i
            ax.hlines(y_bar, min(c_ranks), max(c_ranks), color="black", linewidth=3)
    else:
        c_rank = float(avg_ranks[control])
        ax.axvspan(c_rank - cd, c_rank + cd, color="tab:green", alpha=0.2, label=f"±CD ({control})")

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Posto médio (melhor à direita)")
    if title:
        ax.set_title(title)
    return fig
