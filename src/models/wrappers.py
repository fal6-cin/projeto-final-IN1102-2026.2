"""Wrappers compatíveis com scikit-learn.

`ImputedClassifier` é infraestrutura (usado pelos baselines). `TabDPTWrapper`,
`ModernNCAWrapper` e `TabRWrapper` são esqueletos para os grupos desses modelos
tema, que não têm interface scikit-learn nativa completa ou não aceitam
diretamente os tipos de atributos do projeto.

Todos expõem `fit(X, y)`, `predict_proba(X)`, `predict(X)` e `classes_`, e
aceitam `pandas.DataFrame` com colunas numéricas e categóricas e valores
ausentes. Os imports dos pacotes de cada modelo são feitos sob demanda.
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [
        c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype == bool
    ]
    numerical = [c for c in X.columns if c not in categorical]
    return numerical, categorical


class _LabelEncodingMixin:
    def _encode_labels(self, y) -> np.ndarray:
        self._label_encoder = LabelEncoder().fit(np.asarray(y))
        self.classes_ = self._label_encoder.classes_
        return self._label_encoder.transform(np.asarray(y))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ImputedClassifier(ClassifierMixin, BaseEstimator):
    """Imputa valores ausentes numéricos antes de delegar ao estimador interno.

    Necessário para modelos que não aceitam NaN em colunas numéricas (TabM,
    RealMLP, FT-Transformer e xRFM do pytabkit). Cada coluna numérica com
    ausência no treino recebe a mediana do treino e um indicador binário de
    ausência; colunas categóricas e o tipo `DataFrame` são preservados.
    """

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, median in self._medians.items():
            missing = X[col].isna()
            X[f"{col}__ausente"] = missing.astype(np.float32)
            X[col] = X[col].fillna(median)
        # Ausências em colunas sem ausência no treino recebem a mediana sem indicador.
        for col, median in self._all_medians.items():
            if col not in self._medians:
                X[col] = X[col].fillna(median)
        return X

    def fit(self, X: pd.DataFrame, y) -> ImputedClassifier:
        numerical, _ = _split_columns(X)
        # Coluna inteiramente ausente no treino recebe 0 (mediana indefinida).
        self._all_medians = {c: float(np.nan_to_num(X[c].median())) for c in numerical}
        self._medians = {c: m for c, m in self._all_medians.items() if X[c].isna().any()}
        self.estimator_ = clone(self.estimator).fit(self._transform(X), y)
        self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(self._transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TabDPTWrapper(_LabelEncodingMixin, ClassifierMixin, BaseEstimator):
    """TabDPT-1.3 (Layer 6): in-context learning com recuperação de vizinhos.

    Colunas categóricas são codificadas ordinalmente; valores ausentes são
    tratados pelo próprio TabDPT (indicadores de ausência).
    """

    def __init__(
        self,
        n_ensembles: int = 8,
        context_size: int = 2048,
        temperature: float = 0.8,
        device: str | None = None,
        random_state: int = 42,
    ):
        self.n_ensembles = n_ensembles
        self.context_size = context_size
        self.temperature = temperature
        self.device = device
        self.random_state = random_state

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("TODO(grupo): combine colunas numéricas e categóricas codificadas em um único array float32")

    def fit(self, X: pd.DataFrame, y) -> TabDPTWrapper:
        raise NotImplementedError("TODO(grupo): codifique rótulos e categóricas (categorias novas no teste devem ser aceitas) e treine tabdpt.TabDPTClassifier (fit recebe arrays numpy)")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("TODO(grupo): use ensemble_predict_proba do TabDPT com n_ensembles, temperature, context_size e seed")


class ModernNCAWrapper(_LabelEncodingMixin, ClassifierMixin, BaseEstimator):
    """ModernNCA (Ye et al., ICLR 2025) via TALENT.

    Reserva `val_size` do treino para parada antecipada, como no protocolo do
    TALENT. `config` permite sobrescrever hiperparâmetros do modelo
    (ex.: {"model": {"dim": 128}, "training": {"lr": 1e-3}}).
    """

    def __init__(
        self,
        max_epoch: int = 200,
        batch_size: int = 1024,
        val_size: float = 0.2,
        config: dict[str, Any] | None = None,
        random_state: int = 42,
    ):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.val_size = val_size
        self.config = config
        self.random_state = random_state

    @staticmethod
    def _blocks(X: pd.DataFrame, numerical: list[str], categorical: list[str]):
        raise NotImplementedError("TODO(grupo): separe os blocos numérico (float) e categórico (str) no formato do TALENT; aceite categorias novas e uma única instância")

    def fit(self, X: pd.DataFrame, y) -> ModernNCAWrapper:
        raise NotImplementedError("TODO(grupo): monte (N, C, y) com chaves train/val, o dicionário info e treine a classe do registro 'modernNCA' do TALENT (TALENT.build_args, TALENT.get_method_spec)")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("TODO(grupo): obtenha as probabilidades do modelo treinado com TALENT.model.lib.evaluation.evaluate (tune_threshold=False)")

    def __del__(self):
        path = getattr(self, "_save_path", None)
        if path:
            shutil.rmtree(path, ignore_errors=True)


class TabRWrapper(ClassifierMixin, BaseEstimator):
    """TabR (Gorishniy et al., ICLR 2024) via pytabkit (`TabR_S_D_Classifier`).

    O `TabR_S_D_Classifier` do pytabkit 1.7 não aceita valores ausentes nem, na
    prática, colunas categóricas. O wrapper converte os dados para uma matriz
    numérica: imputação pela mediana do treino com indicadores de ausência e
    codificação one-hot das categóricas (categorias novas no teste devem ser
    aceitas). Hiperparâmetros do TabR (ex.: n_epochs, d_main, context_size, lr)
    são repassados via `params`.
    """

    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42):
        self.params = params
        self.random_state = random_state

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("TODO(grupo): monte a matriz numérica (mediana + indicadores de ausência + one-hot das categóricas)")

    def fit(self, X: pd.DataFrame, y) -> TabRWrapper:
        raise NotImplementedError("TODO(grupo): aprenda medianas e codificador no treino e treine pytabkit.TabR_S_D_Classifier")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("TODO(grupo): transforme X com o que foi aprendido no treino e devolva as probabilidades")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
