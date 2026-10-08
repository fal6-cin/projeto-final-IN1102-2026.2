"""Aceitação do módulo src/pipeline/demsar.py (Demšar, 2006).

Critério objetivo: reproduzir o exemplo do artigo (Tabela 6, Seções 3.1.3 e
3.2.2) e os valores críticos da Tabela 5, e coincidir com scipy quando aplicável.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.pipeline import demsar as D
from tests.aceitacao import demsar2006_tabela6 as T

matplotlib.use("Agg")


def test_rank_matrix_reproduces_published_ranks():
    computed = D.rank_matrix(T.AUC.drop(index="voting"))
    expected = T.RANKS.drop(index="voting").astype(float)
    pd.testing.assert_frame_equal(computed, expected, check_dtype=False)


def test_average_ranks_match_table6():
    avg = D.friedman_from_ranks(T.RANKS).avg_ranks
    for model, expected in T.AVG_RANKS.items():
        assert avg[model] == pytest.approx(expected, abs=1e-3)


def test_friedman_and_iman_davenport_match_paper():
    res = D.friedman_from_ranks(T.RANKS)
    assert res.chi2_f == pytest.approx(T.CHI2_F, abs=0.01)
    assert res.f_f == pytest.approx(T.F_F, abs=0.01)
    assert (res.df1, res.df2) == (3, 39)
    # Valor crítico F(3, 39) a 5% é 2,85: H0 rejeitada.
    assert res.p_value_ff < 0.05


def test_friedman_matches_scipy_without_ties():
    rng = np.random.default_rng(0)
    table = pd.DataFrame(rng.uniform(size=(20, 5)), columns=list("abcde"))
    res = D.friedman_test(table)
    ref = stats.friedmanchisquare(*[table[c] for c in table.columns])
    assert res.chi2_f == pytest.approx(ref.statistic, rel=1e-9)
    assert res.p_value_chi2 == pytest.approx(ref.pvalue, rel=1e-6)


@pytest.mark.parametrize("k", range(2, 11))
def test_q_alpha_table5(k):
    assert D.q_alpha(k, 0.05, "nemenyi") == pytest.approx(T.Q_NEMENYI_005[k - 2], abs=2e-3)
    assert D.q_alpha(k, 0.05, "bonferroni-dunn") == pytest.approx(
        T.Q_BONFERRONI_DUNN_005[k - 2], abs=2e-3
    )


def test_critical_difference_matches_paper():
    assert D.critical_difference(4, 14, 0.05) == pytest.approx(T.CD_NEMENYI_005, abs=0.01)
    assert D.critical_difference(4, 14, 0.10) == pytest.approx(T.CD_NEMENYI_010, abs=0.01)


def test_nemenyi_pairs_at_alpha_010():
    avg = D.friedman_from_ranks(T.RANKS).avg_ranks
    pairs = D.nemenyi_pairs(avg, 14, alpha=0.10)
    significant = {
        frozenset((row.model_a, row.model_b)) for row in pairs.itertuples() if row.significant
    }
    assert set(pairs.columns) >= {"model_a", "model_b", "rank_diff", "cd", "significant"}
    assert len(pairs) == 6
    # Seção 3.2.2 (p. 14): com alpha = 0,10, C4.5 difere de C4.5+m e de C4.5+m+cf.
    assert significant == {frozenset({"C4.5", "C4.5+m"}), frozenset({"C4.5", "C4.5+m+cf"})}


def test_holm_adjustment():
    np.testing.assert_allclose(D.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_holm_vs_control_z_values():
    avg = D.friedman_from_ranks(T.RANKS).avg_ranks
    out = D.holm_vs_control(avg, 14, control="C4.5")
    assert set(out.columns) >= {
        "control",
        "model",
        "rank_diff",
        "z",
        "p_value",
        "p_holm",
        "significant",
    }
    out = out.set_index("model")
    for model, z in T.Z_VS_C45.items():
        assert abs(out.loc[model, "z"]) == pytest.approx(z, abs=0.01)
    assert out.loc["C4.5+m+cf", "p_holm"] <= out.loc["C4.5+cf", "p_holm"]


def test_wilcoxon_matches_paper_example():
    res = D.wilcoxon_signed_ranks(T.AUC["C4.5+m"], T.AUC["C4.5"])
    assert set(res) >= {"n", "r_plus", "r_minus", "T", "p_value", "wins", "ties", "losses"}
    assert (res["wins"], res["ties"], res["losses"]) == (10, 2, 2)
    assert res["r_plus"] == pytest.approx(T.WILCOXON["r_plus"])
    assert res["r_minus"] == pytest.approx(T.WILCOXON["r_minus"])
    assert res["T"] == pytest.approx(T.WILCOXON["T"])
    assert res["p_value"] < 0.05


def test_wilcoxon_vs_control_schema():
    out = D.wilcoxon_vs_control(T.AUC, "C4.5+m")
    assert set(out.columns) >= {"control", "model", "r_plus", "p_value", "p_holm", "significant"}
    assert set(out["model"]) == {"C4.5", "C4.5+cf", "C4.5+m+cf"}
    assert (out["p_holm"] >= out["p_value"]).all()


def test_friedman_total_agreement_gives_infinite_ff():
    table = pd.DataFrame({"a": [0.9, 0.8, 0.7, 0.6], "b": [0.5, 0.4, 0.3, 0.2]})
    res = D.friedman_test(table)
    assert np.isinf(res.f_f) and res.p_value_ff == 0.0


def test_lower_is_better_inverts_ranks():
    table = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.5, 0.6, 0.7]})
    assert D.average_ranks(table, higher_is_better=False).to_dict() == {"a": 1.0, "b": 2.0}
    assert D.average_ranks(table, higher_is_better=True).to_dict() == {"a": 2.0, "b": 1.0}


def test_cd_cliques_and_diagram():
    avg = D.friedman_from_ranks(T.RANKS).avg_ranks
    cd = D.critical_difference(4, 14, 0.10)
    cliques = [set(c) for c in D.cd_cliques(avg, cd)]
    assert {"C4.5+m+cf", "C4.5+m", "C4.5+cf"} in cliques
    assert {"C4.5+cf", "C4.5"} in cliques
    assert D.plot_cd_diagram(avg, cd) is not None
