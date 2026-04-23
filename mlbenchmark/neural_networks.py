"""
neural_networks.py  ·  5 Tipos de Redes Neuronales
BCD-7213 Minería de Datos Avanzada - Universidad LEAD
Caso de Estudio 2  ·  I Cuatrimestre 2026

Arquitecturas implementadas
──────────────────────────────────────────────────────────────
1. MLP sklearn         — MLPClassifier/MLPRegressor (scikit-learn)
2. Feedforward Keras   — Dense layers básicas (TensorFlow/Keras)
3. Deep Dropout Keras  — Capas profundas + Dropout + BatchNorm
4. CNN 1D Keras        — Conv1D sobre datos tabulares reshaped
5. Autoencoder Keras   — Encoder-Decoder (detección de anomalías)

Todas las clases exponen la misma interfaz sklearn-compatible:
    .fit(X, y)
    .predict(X)
    .predict_proba(X)  ← clasificadores supervisados
    .get_metrics(X_test, y_test) → dict

Función de conveniencia:
    benchmark_neural_networks(X_train, X_test, y_train, y_test,
                               task='classification') → pd.DataFrame
"""

from __future__ import annotations

import abc
import contextlib
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────
# UTILIDADES INTERNAS
# ─────────────────────────────────────────────────────────────────

def _check_keras() -> bool:
    """Verifica que TensorFlow/Keras esté disponible en el entorno."""
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except ImportError:
        return False


def _set_global_seed(seed: int) -> None:
    """Fija semillas en numpy y tensorflow para reproducibilidad."""
    np.random.seed(seed)
    with contextlib.suppress(Exception):
        import tensorflow as tf
        tf.random.set_seed(seed)


def _validate_input(X: np.ndarray, name: str = "X") -> np.ndarray:
    """
    Convierte a float64, verifica NaN e infinitos.

    Args:
        X: Array de entrada.
        name: Nombre para mensajes de error.

    Returns:
        Array limpio como np.ndarray float64.

    Raises:
        ValueError: Si X contiene NaN o infinitos.
    """
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError(
            f"'{name}' contiene valores NaN o infinitos. "
            "Limpia los datos antes de entrenar."
        )
    return X


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    """
    Calcula métricas de clasificación.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones de clase.
        y_prob: Probabilidades por clase (opcional, para AUC-ROC).

    Returns:
        Dict con Accuracy, F1-Score y AUC-ROC.
    """
    metrics = {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "F1-Score": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
    }
    if y_prob is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
            metrics["AUC-ROC"] = round(float(auc), 4)
        except Exception:
            metrics["AUC-ROC"] = None
    else:
        metrics["AUC-ROC"] = None
    return metrics


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula métricas de regresión.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.

    Returns:
        Dict con R², RMSE y MAE.
    """
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R²":   round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def _get_keras_callbacks(patience: int = 10):
    """
    Devuelve los callbacks estándar para entrenamiento Keras.

    Incluye EarlyStopping (detiene si val_loss no mejora) y
    ReduceLROnPlateau (reduce lr al estancarse).
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=0,
        ),
    ]


# ─────────────────────────────────────────────────────────────────
# CLASE BASE ABSTRACTA
# ─────────────────────────────────────────────────────────────────

class BaseNeuralModel(abc.ABC):
    """
    Interfaz común para todos los modelos de redes neuronales.

    Subclases deben implementar `_build()`.
    El atributo `is_supervised` diferencia modelos supervisados
    (clasificación/regresión) de no supervisados (autoencoder).
    """

    NAME: str = "BaseModel"
    is_supervised: bool = True

    def __init__(
        self,
        task: str = "classification",
        random_state: int = 42,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task debe ser 'classification' o 'regression', no '{task}'."
            )
        self.task = task
        self.random_state = random_state
        self._fitted = False
        self._le = LabelEncoder()

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseNeuralModel":
        """Entrena el modelo. Retorna self para encadenamiento."""

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Genera predicciones sobre X."""

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Probabilidades por clase. Solo disponible en clasificación."""
        del X  # parámetro de interfaz; subclases lo usan en su implementación
        return None

    def get_metrics(
        self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None
    ) -> dict:
        """Calcula métricas sobre el conjunto de test."""
        self._check_fitted()
        y_pred = self.predict(X_test)
        if self.task == "classification":
            return _classification_metrics(y_test, y_pred, self.predict_proba(X_test))
        return _regression_metrics(y_test, y_pred)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(
                f"{self.__class__.__name__} no ha sido entrenado. Llama a .fit() primero."
            )

    @property
    def history(self) -> dict:
        """Historial de entrenamiento (vacío en modelos sin Keras)."""
        return {}


# ─────────────────────────────────────────────────────────────────
# CLASE BASE PARA MODELOS KERAS SUPERVISADOS
# ─────────────────────────────────────────────────────────────────

class BaseKerasModel(BaseNeuralModel):
    """
    Base compartida para todos los modelos Keras supervisados.

    Centraliza la lógica de fit/predict/predict_proba eliminando
    la duplicación entre FeedforwardKerasModel, DeepDropoutModel y CNN1DModel.

    Subclases solo necesitan implementar `_build(n_features, n_classes)`.
    `_prepare_X(X)` puede sobreescribirse para transformaciones de entrada
    (ej: reshape 3D para CNN).
    """

    is_supervised = True

    def __init__(
        self,
        task: str = "classification",
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        random_state: int = 42,
    ) -> None:
        super().__init__(task=task, random_state=random_state)
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self._model = None
        self._history = None
        self._n_classes: int = 0

    @abc.abstractmethod
    def _build(self, n_features: int, n_classes: int):
        """Construye y compila el modelo Keras. Retorna model."""

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        """Hook para transformar X antes de pasarlo al modelo (ej: reshape 3D)."""
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseKerasModel":
        """
        Entrena el modelo Keras con EarlyStopping y ReduceLROnPlateau.

        Args:
            X: Features de entrenamiento (ya escalados, sin NaN).
            y: Target de entrenamiento.

        Returns:
            self
        """
        _set_global_seed(self.random_state)
        X = _validate_input(X, "X")

        if self.task == "classification":
            y_enc = self._le.fit_transform(y)
            self._n_classes = len(self._le.classes_)
        else:
            y_enc = np.asarray(y, dtype=np.float64)
            self._n_classes = 1

        self._model = self._build(X.shape[1], self._n_classes)
        X_in = self._prepare_X(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._history = self._model.fit(
                X_in,
                y_enc,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=_get_keras_callbacks(patience=max(5, self.epochs // 10)),
                verbose=0,
            )

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = _validate_input(X, "X")
        raw = self._model.predict(self._prepare_X(X), verbose=0)

        if self.task == "classification":
            if self._n_classes == 2:
                y_enc = (raw[:, 0] >= 0.5).astype(int)
            else:
                y_enc = np.argmax(raw, axis=1)
            return self._le.inverse_transform(y_enc)
        return raw[:, 0]

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.task != "classification":
            return None
        self._check_fitted()
        X = _validate_input(X, "X")
        raw = self._model.predict(self._prepare_X(X), verbose=0)
        if self._n_classes == 2:
            return np.column_stack([1 - raw[:, 0], raw[:, 0]])
        return raw

    @property
    def history(self) -> dict:
        return self._history.history if self._history else {}

    # ── Helpers de compilación reutilizables ─────────────────────

    def _compile_output(self, model, n_classes: int):
        """
        Agrega la capa de salida y compila según task y n_classes.

        Args:
            model: Modelo Keras al que añadir la salida.
            n_classes: Número de clases (1 para regresión o binario).
        """
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam

        if self.task == "classification":
            if n_classes == 2:
                model.add(Dense(1, activation="sigmoid"))
                model.compile(
                    optimizer=Adam(learning_rate=1e-3),
                    loss="binary_crossentropy",
                    metrics=["accuracy"],
                )
            else:
                model.add(Dense(n_classes, activation="softmax"))
                model.compile(
                    optimizer=Adam(learning_rate=1e-3),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss="mse",
                metrics=["mae"],
            )
        return model


# ─────────────────────────────────────────────────────────────────
# 1. MLP SKLEARN
# ─────────────────────────────────────────────────────────────────

class MLPSklearnModel(BaseNeuralModel):
    """
    Perceptrón Multicapa usando scikit-learn.

    Architecture: Input → Dense(64) → Dense(32) → Dense(16) → Output
    Activation  : ReLU (ocultas) · Softmax/Identidad (salida)
    Optimizer   : Adam

    Args:
        task: 'classification' o 'regression'.
        hidden_layers: Tupla con el tamaño de cada capa oculta.
        activation: Función de activación ('relu', 'tanh', 'logistic').
        solver: Optimizador sklearn ('adam', 'lbfgs', 'sgd').
        max_iter: Iteraciones máximas de entrenamiento.
        random_state: Semilla de reproducibilidad.
    """

    NAME = "MLP (sklearn)"

    def __init__(
        self,
        task: str = "classification",
        hidden_layers: Tuple[int, ...] = (64, 32, 16),
        activation: str = "relu",
        solver: str = "adam",
        max_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        super().__init__(task=task, random_state=random_state)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPSklearnModel":
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        X = _validate_input(X, "X")
        _set_global_seed(self.random_state)

        common_kwargs = dict(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            solver=self.solver,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

        if self.task == "classification":
            y_enc = self._le.fit_transform(y)
            self._model = MLPClassifier(**common_kwargs)
            self._model.fit(X, y_enc)
        else:
            self._model = MLPRegressor(**common_kwargs)
            self._model.fit(X, np.asarray(y, dtype=np.float64))

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = _validate_input(X, "X")
        raw = self._model.predict(X)
        if self.task == "classification":
            return self._le.inverse_transform(raw)
        return raw

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.task != "classification":
            return None
        self._check_fitted()
        return self._model.predict_proba(_validate_input(X, "X"))


# ─────────────────────────────────────────────────────────────────
# 2. FEEDFORWARD KERAS (Red Profunda Básica)
# ─────────────────────────────────────────────────────────────────

class FeedforwardKerasModel(BaseKerasModel):
    """
    Red Neuronal Feedforward básica con TensorFlow/Keras.

    Architecture: Input → Dense(128,ReLU) → Dense(64,ReLU) → Dense(32,ReLU) → Output
    Optimizer   : Adam (lr=1e-3)

    Args:
        task: 'classification' o 'regression'.
        epochs: Épocas máximas (EarlyStopping puede detener antes).
        batch_size: Tamaño de mini-batch.
        validation_split: Fracción de datos para validación interna.
        random_state: Semilla.
    """

    NAME = "Feedforward (Keras)"

    def _build(self, n_features: int, n_classes: int):
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential

        model = Sequential(name="Feedforward")
        model.add(Dense(128, activation="relu", input_shape=(n_features,)))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        return self._compile_output(model, n_classes)


# ─────────────────────────────────────────────────────────────────
# 3. DEEP DROPOUT (Red Profunda con Regularización)
# ─────────────────────────────────────────────────────────────────

class DeepDropoutModel(BaseKerasModel):
    """
    Red Neuronal Profunda con Dropout y BatchNormalization.

    Architecture:
        Dense(256,ReLU) → BN → Dropout(drop+0.1)
        Dense(128,ReLU) → BN → Dropout(drop)
        Dense(64,ReLU)  → BN → Dropout(max(0.1, drop-0.1))
        Output
    Optimizer   : Adam (lr=1e-3)
    Regularizer : L2(1e-4) + Dropout + BatchNorm

    Args:
        task: 'classification' o 'regression'.
        epochs: Épocas máximas.
        batch_size: Tamaño de mini-batch.
        dropout_rate: Tasa de dropout base (se ajusta por capa).
        validation_split: Fracción de datos para validación.
        random_state: Semilla.
    """

    NAME = "Deep + Dropout (Keras)"

    def __init__(
        self,
        task: str = "classification",
        epochs: int = 60,
        batch_size: int = 32,
        dropout_rate: float = 0.3,
        validation_split: float = 0.1,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            task=task,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            random_state=random_state,
        )
        self.dropout_rate = dropout_rate

    def _build(self, n_features: int, n_classes: int):
        from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.regularizers import l2

        dr = self.dropout_rate
        model = Sequential(name="DeepDropout")
        model.add(
            Dense(256, activation="relu", kernel_regularizer=l2(1e-4),
                  input_shape=(n_features,))
        )
        model.add(BatchNormalization())
        model.add(Dropout(min(0.9, dr + 0.1)))

        model.add(Dense(128, activation="relu", kernel_regularizer=l2(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(dr))

        model.add(Dense(64, activation="relu", kernel_regularizer=l2(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(max(0.1, dr - 0.1)))

        return self._compile_output(model, n_classes)


# ─────────────────────────────────────────────────────────────────
# 4. CNN 1D (Convolucional sobre datos tabulares)
# ─────────────────────────────────────────────────────────────────

class CNN1DModel(BaseKerasModel):
    """
    Red Neuronal Convolucional 1D sobre datos tabulares.

    Los features se tratan como una secuencia de longitud n_features.
    Architecture:
        Input(n,1) → Conv1D(filters, k, ReLU) → Conv1D(filters//2, k, ReLU)
                   → GlobalAvgPool1D → Dense(64, ReLU) → Dropout(0.2) → Output

    Args:
        task: 'classification' o 'regression'.
        epochs: Épocas máximas.
        batch_size: Tamaño de mini-batch.
        filters: Número de filtros en la primera capa convolucional.
        kernel_size: Tamaño del kernel convolucional.
        validation_split: Fracción de datos para validación.
        random_state: Semilla.
    """

    NAME = "CNN 1D (Keras)"

    def __init__(
        self,
        task: str = "classification",
        epochs: int = 50,
        batch_size: int = 32,
        filters: int = 64,
        kernel_size: int = 3,
        validation_split: float = 0.1,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            task=task,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            random_state=random_state,
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self._n_features: int = 0

    def _prepare_X(self, X: np.ndarray) -> np.ndarray:
        """Reshape a (N, features, 1) para Conv1D."""
        return X.reshape(len(X), X.shape[1], 1)

    def _build(self, n_features: int, n_classes: int):
        from tensorflow.keras.layers import (
            Conv1D, Dense, Dropout, GlobalAveragePooling1D,
        )
        from tensorflow.keras.models import Sequential

        # Kernel size no puede superar la longitud de la secuencia
        ks = min(self.kernel_size, max(1, n_features - 1))
        self._n_features = n_features

        model = Sequential(name="CNN1D")
        model.add(
            Conv1D(self.filters, ks, activation="relu",
                   input_shape=(n_features, 1), padding="same")
        )
        model.add(Conv1D(self.filters // 2, ks, activation="relu", padding="same"))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))
        return self._compile_output(model, n_classes)


# ─────────────────────────────────────────────────────────────────
# 5. AUTOENCODER KERAS (Detección de Anomalías)
# ─────────────────────────────────────────────────────────────────

class AutoencoderModel(BaseNeuralModel):
    """
    Autoencoder no supervisado para detección de anomalías.

    Architecture:
        Encoder: Input → Dense(64,ReLU) → Dense(32,ReLU) → Dense(latent,ReLU)
        Decoder: Latent → Dense(32,ReLU) → Dense(64,ReLU) → Dense(n_features,Linear)

    El error de reconstrucción por muestra se usa como score de anomalía.
    Las muestras con error > percentil(threshold_pct) en entrenamiento
    se clasifican como anomalías.

    Args:
        latent_dim: Dimensión del espacio latente (cuello de botella).
        epochs: Épocas máximas de entrenamiento.
        batch_size: Tamaño de mini-batch.
        threshold_pct: Percentil sobre datos de train para definir el umbral.
        validation_split: Fracción interna para validación durante entrenamiento.
        random_state: Semilla.
    """

    NAME = "Autoencoder (Keras)"
    is_supervised = False

    def __init__(
        self,
        latent_dim: int = 8,
        epochs: int = 80,
        batch_size: int = 32,
        threshold_pct: float = 90.0,
        validation_split: float = 0.1,
        random_state: int = 42,
    ) -> None:
        # task="classification" es placeholder; Autoencoder ignora y en fit
        super().__init__(task="classification", random_state=random_state)
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_pct = threshold_pct
        self.validation_split = validation_split
        self._autoencoder = None
        self._encoder = None
        self._threshold: float = 0.0
        self._history = None

    def _build(self, n_features: int):
        from tensorflow.keras.layers import Dense, Input
        from tensorflow.keras.models import Model

        inp = Input(shape=(n_features,), name="input")
        x = Dense(64, activation="relu", name="enc_1")(inp)
        x = Dense(32, activation="relu", name="enc_2")(x)
        latent = Dense(self.latent_dim, activation="relu", name="latent")(x)

        x = Dense(32, activation="relu", name="dec_1")(latent)
        x = Dense(64, activation="relu", name="dec_2")(x)
        out = Dense(n_features, activation="linear", name="output")(x)

        autoencoder = Model(inputs=inp, outputs=out, name="Autoencoder")
        autoencoder.compile(optimizer="adam", loss="mse")

        encoder = Model(inputs=inp, outputs=latent, name="Encoder")
        return autoencoder, encoder

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AutoencoderModel":
        """
        Entrena el autoencoder en modo no supervisado (ignora y).

        Args:
            X: Features de entrenamiento.
            y: Ignorado (interfaz sklearn-compatible).

        Returns:
            self
        """
        _set_global_seed(self.random_state)
        X = _validate_input(X, "X")
        self._autoencoder, self._encoder = self._build(X.shape[1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._history = self._autoencoder.fit(
                X, X,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=_get_keras_callbacks(patience=max(5, self.epochs // 10)),
                verbose=0,
            )

        # Umbral de anomalía basado en los datos de entrenamiento
        errors = self._reconstruction_error_raw(X)
        self._threshold = float(np.percentile(errors, self.threshold_pct))
        self._fitted = True
        return self

    def _reconstruction_error_raw(self, X: np.ndarray) -> np.ndarray:
        recon = self._autoencoder.predict(X, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Error cuadrático medio de reconstrucción por muestra."""
        self._check_fitted()
        return self._reconstruction_error_raw(_validate_input(X, "X"))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Clasifica cada muestra como anomalía (1) o normal (0).

        Args:
            X: Features a evaluar.

        Returns:
            Array binario: 1=anomalía, 0=normal.
        """
        return (self.reconstruction_error(X) > self._threshold).astype(int)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Retorna la representación latente (embedding) de los datos."""
        self._check_fitted()
        return self._encoder.predict(_validate_input(X, "X"), verbose=0)

    def get_metrics(
        self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None
    ) -> dict:
        """
        Métricas del autoencoder sobre el conjunto de test.

        Args:
            X_test: Features de test.
            y_test: Etiquetas de anomalía 0/1 opcionales (para Accuracy/F1).

        Returns:
            Dict con error de reconstrucción, threshold y estadísticas de anomalías.
        """
        self._check_fitted()
        errors = self.reconstruction_error(X_test)
        preds = (errors > self._threshold).astype(int)

        metrics = {
            "Error Reconstrucción (medio)": round(float(errors.mean()), 6),
            "Error Reconstrucción (max)":   round(float(errors.max()), 6),
            "Threshold":                    round(self._threshold, 6),
            "Anomalías detectadas":         int(preds.sum()),
            "% Anomalías":                  round(float(preds.mean() * 100), 2),
        }
        if y_test is not None:
            with contextlib.suppress(Exception):
                metrics["Accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
                metrics["F1-Score"] = round(
                    float(f1_score(y_test, preds, zero_division=0)), 4
                )
        return metrics

    @property
    def history(self) -> dict:
        return self._history.history if self._history else {}


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN DE BENCHMARKING INTEGRADO
# ─────────────────────────────────────────────────────────────────

def benchmark_neural_networks(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str = "classification",
    epochs: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Entrena y evalúa los 5 tipos de redes neuronales sobre los mismos datos.

    Args:
        X_train: Features de entrenamiento (recomendado: ya escalados).
        X_test:  Features de test.
        y_train: Target de entrenamiento.
        y_test:  Target de test.
        task:    'classification' | 'regression'.
        epochs:  Épocas máximas para modelos Keras (EarlyStopping puede detener antes).
        random_state: Semilla global de reproducibilidad.

    Returns:
        pd.DataFrame con métricas por arquitectura, ordenado por columna Modelo.
    """
    keras_ok = _check_keras()
    X_train = _validate_input(np.asarray(X_train), "X_train")
    X_test  = _validate_input(np.asarray(X_test),  "X_test")

    _common_keras = dict(task=task, epochs=epochs, random_state=random_state)

    models: list[BaseNeuralModel] = [
        MLPSklearnModel(task=task, random_state=random_state),
    ]

    if keras_ok:
        models += [
            FeedforwardKerasModel(**_common_keras),
            DeepDropoutModel(**_common_keras),
            CNN1DModel(**_common_keras),
            AutoencoderModel(epochs=epochs, random_state=random_state),
        ]
    else:
        # Fallback: variantes de MLP sklearn con distintas configuraciones
        models += [
            MLPSklearnModel(task=task, hidden_layers=(128, 64, 32),
                            activation="tanh", random_state=random_state),
            MLPSklearnModel(task=task, hidden_layers=(256, 128),
                            activation="relu", solver="lbfgs",
                            random_state=random_state),
            MLPSklearnModel(task=task, hidden_layers=(64, 64, 64),
                            activation="relu", random_state=random_state),
            MLPSklearnModel(task=task, hidden_layers=(32, 16),
                            activation="logistic", random_state=random_state),
        ]

    results = []

    for model in models:
        row: dict = {"Modelo": model.NAME}
        try:
            if model.is_supervised:
                model.fit(X_train, y_train)
                metrics = model.get_metrics(X_test, y_test)
                row["Tipo"] = "Supervisado"
            else:
                model.fit(X_train)
                metrics = model.get_metrics(X_test)
                row["Tipo"] = "No Supervisado"

            row.update(metrics)

        except Exception as exc:
            row["Tipo"] = "Error"
            row["_error"] = str(exc)

        results.append(row)

    df = pd.DataFrame(results)
    first_cols = ["Modelo", "Tipo"]
    other_cols = [c for c in df.columns if c not in first_cols]
    return df[first_cols + other_cols]


# ─────────────────────────────────────────────────────────────────
# RESUMEN DE ARQUITECTURAS (para la UI de Streamlit)
# ─────────────────────────────────────────────────────────────────

ARCHITECTURES_INFO = {
    "MLP (sklearn)": {
        "Tipo":           "Perceptrón Multicapa",
        "Framework":      "scikit-learn",
        "Capas":          "Input → Dense(64) → Dense(32) → Dense(16) → Output",
        "Activación":     "ReLU (ocultas) · Softmax/Lineal (salida)",
        "Optimizer":      "Adam",
        "Regularización": "Ninguna",
        "Uso":            "Referencia base, rápido y robusto",
    },
    "Feedforward (Keras)": {
        "Tipo":           "Red Densa Básica",
        "Framework":      "TensorFlow / Keras",
        "Capas":          "Input → Dense(128,ReLU) → Dense(64,ReLU) → Dense(32,ReLU) → Output",
        "Activación":     "ReLU (ocultas) · Sigmoid/Softmax/Lineal (salida)",
        "Optimizer":      "Adam (lr=1e-3) + EarlyStopping + ReduceLROnPlateau",
        "Regularización": "Ninguna",
        "Uso":            "Línea base con Keras, más capas que MLP sklearn",
    },
    "Deep + Dropout (Keras)": {
        "Tipo":           "Red Profunda Regularizada",
        "Framework":      "TensorFlow / Keras",
        "Capas":          "Dense(256) → BN → Drop → Dense(128) → BN → Drop → Dense(64) → BN → Drop → Output",
        "Activación":     "ReLU + BatchNormalization",
        "Optimizer":      "Adam (lr=1e-3) + EarlyStopping + ReduceLROnPlateau",
        "Regularización": "L2(1e-4) + Dropout(0.4/0.3/0.2) + BatchNorm",
        "Uso":            "Previene overfitting en datasets pequeños/medianos",
    },
    "CNN 1D (Keras)": {
        "Tipo":           "Red Convolucional 1D",
        "Framework":      "TensorFlow / Keras",
        "Capas":          "Conv1D(64,k=3) → Conv1D(32,k=3) → GlobalAvgPool → Dense(64) → Output",
        "Activación":     "ReLU",
        "Optimizer":      "Adam (lr=1e-3) + EarlyStopping + ReduceLROnPlateau",
        "Regularización": "Dropout(0.2)",
        "Uso":            "Captura patrones locales en la secuencia de features",
    },
    "Autoencoder (Keras)": {
        "Tipo":           "Autoencoder (No Supervisado)",
        "Framework":      "TensorFlow / Keras",
        "Capas":          "Dense(64)→Dense(32)→Dense(latent=8) | Dense(32)→Dense(64)→Output",
        "Activación":     "ReLU (encoder/decoder) · Lineal (reconstrucción)",
        "Optimizer":      "Adam + EarlyStopping + ReduceLROnPlateau",
        "Regularización": "Cuello de botella latente",
        "Uso":            "Detección de anomalías / reducción dimensional",
    },
}
