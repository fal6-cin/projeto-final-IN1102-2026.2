"""Dados da Tabela 6 de Demšar (2006): AUC de quatro variantes do C4.5 em 14 datasets.

Fonte: Demšar, J. (2006). JMLR 7:1-30, p. 14. Os postos publicados (RANKS) são a
referência para o teste de Friedman. Na linha 'voting' os valores impressos estão
arredondados (0,975 e 0,975), mas os postos publicados são 2 e 3; por isso ela é
excluída da conferência dos postos calculados a partir dos escores.
"""

import pandas as pd

MODELS = ["C4.5", "C4.5+m", "C4.5+cf", "C4.5+m+cf"]
DATASETS = [
    "adult (sample)",
    "breast cancer",
    "breast cancer wisconsin",
    "cmc",
    "ionosphere",
    "iris",
    "liver disorders",
    "lung cancer",
    "lymphography",
    "mushroom",
    "primary tumor",
    "rheum",
    "voting",
    "wine",
]
AUC = pd.DataFrame(
    [
        [0.763, 0.768, 0.771, 0.798],
        [0.599, 0.591, 0.590, 0.569],
        [0.954, 0.971, 0.968, 0.967],
        [0.628, 0.661, 0.654, 0.657],
        [0.882, 0.888, 0.886, 0.898],
        [0.936, 0.931, 0.916, 0.931],
        [0.661, 0.668, 0.609, 0.685],
        [0.583, 0.583, 0.563, 0.625],
        [0.775, 0.838, 0.866, 0.875],
        [1.000, 1.000, 1.000, 1.000],
        [0.940, 0.962, 0.965, 0.962],
        [0.619, 0.666, 0.614, 0.669],
        [0.972, 0.981, 0.975, 0.975],
        [0.957, 0.978, 0.946, 0.970],
    ],
    index=DATASETS,
    columns=MODELS,
)
RANKS = pd.DataFrame(
    [
        [4, 3, 2, 1],
        [1, 2, 3, 4],
        [4, 1, 2, 3],
        [4, 1, 3, 2],
        [4, 2, 3, 1],
        [1, 2.5, 4, 2.5],
        [3, 2, 4, 1],
        [2.5, 2.5, 4, 1],
        [4, 3, 2, 1],
        [2.5, 2.5, 2.5, 2.5],
        [4, 2.5, 1, 2.5],
        [3, 2, 4, 1],
        [4, 1, 2, 3],
        [3, 1, 4, 2],
    ],
    index=DATASETS,
    columns=MODELS,
)
AVG_RANKS = {"C4.5": 3.143, "C4.5+m": 2.000, "C4.5+cf": 2.893, "C4.5+m+cf": 1.964}
CHI2_F = 9.28
F_F = 3.69
CD_NEMENYI_005 = 1.25
CD_NEMENYI_010 = 1.12
# Holm com controle C4.5 (p. 15): z de C4.5+m+cf, C4.5+m e C4.5+cf.
Z_VS_C45 = {"C4.5+m+cf": 2.416, "C4.5+m": 2.342, "C4.5+cf": 0.512}
# Wilcoxon C4.5+m contra C4.5 (Seção 3.1.3): R+ = 93, R- = 12, T = 12.
WILCOXON = {"r_plus": 93.0, "r_minus": 12.0, "T": 12.0}
# Tabela 5 (alpha = 0,05). Para Bonferroni-Dunn com k = 9 o artigo imprime 2,724;
# o valor correto, norm.ppf(1 - 0,05/16), é 2,734 (provável erro tipográfico).
Q_NEMENYI_005 = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]
Q_BONFERRONI_DUNN_005 = [1.960, 2.241, 2.394, 2.498, 2.576, 2.638, 2.690, 2.734, 2.773]
