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
    .predict_proba(X)  ← clasificadores
    .get_metrics(X_test, y_test) → dict

Función de conveniencia:
    benchmark_neural_networks(X_train, X_test, y_train, y_test,
                               task='classification') → pd.DataFrame
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              r2_score, mean_squared_error, mean_absolute_error)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# UTILIDADES INTERNAS
# ─────────────────────────────────────────────────────────────────

def _check_keras():
    """Verifica que TensorFlow/Keras esté disponible."""
    try:
        import tensorflow as tf  # noqa
        return True
    except ImportError:
        return False


def _classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    met = {
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "F1-Score":  round(float(f1_score(y_true, y_pred,
                                          average="weighted", zero_division=0)), 4),
    }
    if y_prob is not None:
        try:
            classes = np.unique(y_true)
            if len(classes) == 2:
                met["AUC-ROC"] = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
            else:
                met["AUC-ROC"] = round(float(roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted")), 4)
        except Exception:
            met["AUC-ROC"] = None
    else:
        met["AUC-ROC"] = None
    return met


def _regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R²":   round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


# ─────────────────────────────────────────────────────────────────
# 1. MLP SKLEARN
# ─────────────────────────────────────────────────────────────────

class MLPSklearnModel:
    """
    Red Neuronal Perceptrón Multicapa usando scikit-learn.

    Architecture: Input → Dense(64) → Dense(32) → Dense(16) → Output
    Activation  : ReLU (hidden) · Softmax/identity (output)
    Optimizer   : adam
    """

    NAME = "MLP (sklearn)"

    def __init__(self, task: str = "classification", hidden_layers=(64, 32, 16),
                 activation: str = "relu", solver: str = "adam",
                 max_iter: int = 500, random_state: int = 42):
        self.task           = task
        self.hidden_layers  = hidden_layers
        self.activation     = activation
        self.solver         = solver
        self.max_iter       = max_iter
        self.random_state   = random_state
        self._model         = None
        self._le            = LabelEncoder()
        self._fitted        = False

    def fit(self, X, y):
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        X = np.asarray(X, dtype=float)
        if self.task == "classification":
            y_enc = self._le.fit_transform(y)
            self._model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layers,
                activation=self.activation,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self._model.fit(X, y_enc)
        else:
            self._model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                activation=self.activation,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self._model.fit(X, np.asarray(y, dtype=float))
        self._fitted = True
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.task == "classification":
            return self._le.inverse_transform(self._model.predict(X))
        return self._model.predict(X)

    def predict_proba(self, X):
        if self.task != "classification":
            return None
        return self._model.predict_proba(np.asarray(X, dtype=float))

    def get_metrics(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        if self.task == "classification":
            y_prob = self.predict_proba(X_test)
            return _classification_metrics(y_test, y_pred, y_prob)
        return _regression_metrics(y_test, y_pred)


# ─────────────────────────────────────────────────────────────────
# 2. FEEDFORWARD KERAS (Red Profunda Básica)
# ─────────────────────────────────────────────────────────────────

class FeedforwardKerasModel:
    """
    Red Neuronal Feedforward usando TensorFlow/Keras.

    Architecture: Input → Dense(128,ReLU) → Dense(64,ReLU) → Dense(32,ReLU) → Output
    Optimizer   : Adam
    Loss        : binary_crossentropy | sparse_categorical_crossentropy | mse
    """

    NAME = "Feedforward (Keras)"

    def __init__(self, task: str = "classification", epochs: int = 50,
                 batch_size: int = 32, random_state: int = 42):
        self.task         = task
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.random_state = random_state
        self._model       = None
        self._le          = LabelEncoder()
        self._history     = None

    def _build(self, n_features: int, n_classes: int):
        import tensorflow as tf
        tf.random.set_seed(self.random_state)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

        model = Sequential(name="Feedforward")
        model.add(Dense(128, activation="relu", input_shape=(n_features,)))
        model.add(Dense(64,  activation="relu"))
        model.add(Dense(32,  activation="relu"))

        if self.task == "classification":
            if n_classes == 2:
                model.add(Dense(1, activation="sigmoid"))
                model.compile(optimizer="adam", loss="binary_crossentropy",
                              metrics=["accuracy"])
            else:
                model.add(Dense(n_classes, activation="softmax"))
                model.compile(optimizer="adam",
                              loss="sparse_categorical_crossentropy",
                              metrics=["accuracy"])
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return model

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if self.task == "classification":
            y_enc = self._le.fit_transform(y)
            n_classes = len(self._le.classes_)
            self._model = self._build(X.shape[1], n_classes)
            if n_classes == 2:
                self._history = self._model.fit(
                    X, y_enc, epochs=self.epochs,
                    batch_size=self.batch_size, verbose=0)
            else:
                self._history = self._model.fit(
                    X, y_enc, epochs=self.epochs,
                    batch_size=self.batch_size, verbose=0)
        else:
            y_f = np.asarray(y, dtype=float)
            self._model = self._build(X.shape[1], 1)
            self._history = self._model.fit(
                X, y_f, epochs=self.epochs,
                batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        raw = self._model.predict(X, verbose=0)
        if self.task == "classification":
            n_classes = len(self._le.classes_)
            if n_classes == 2:
                y_enc = (raw[:, 0] >= 0.5).astype(int)
            else:
                y_enc = np.argmax(raw, axis=1)
            return self._le.inverse_transform(y_enc)
        return raw[:, 0]

    def predict_proba(self, X):
        if self.task != "classification":
            return None
        X = np.asarray(X, dtype=float)
        raw = self._model.predict(X, verbose=0)
        n_classes = len(self._le.classes_)
        if n_classes == 2:
            return np.column_stack([1 - raw[:, 0], raw[:, 0]])
        return raw

    def get_metrics(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        if self.task == "classification":
            return _classification_metrics(y_test, y_pred, self.predict_proba(X_test))
        return _regression_metrics(y_test, y_pred)

    @property
    def history(self):
        return self._history.history if self._history else {}


# ─────────────────────────────────────────────────────────────────
# 3. DEEP DROPOUT (Red Profunda con Regularización)
# ─────────────────────────────────────────────────────────────────

class DeepDropoutModel:
    """
    Red Neuronal Profunda con Dropout y BatchNormalization.

    Architecture: Input → Dense(256,ReLU) → BN → Dropout(0.4)
                         → Dense(128,ReLU) → BN → Dropout(0.3)
                         → Dense(64,ReLU)  → BN → Dropout(0.2)
                         → Output
    Optimizer   : Adam (lr=1e-3)
    Regularizer : L2 + Dropout
    """

    NAME = "Deep + Dropout (Keras)"

    def __init__(self, task: str = "classification", epochs: int = 60,
                 batch_size: int = 32, dropout_rate: float = 0.3,
                 random_state: int = 42):
        self.task         = task
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self._model       = None
        self._le          = LabelEncoder()
        self._history     = None

    def _build(self, n_features: int, n_classes: int):
        import tensorflow as tf
        tf.random.set_seed(self.random_state)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.regularizers import l2

        dr = self.dropout_rate
        model = Sequential(name="DeepDropout")
        model.add(Dense(256, activation="relu", kernel_regularizer=l2(1e-4),
                        input_shape=(n_features,)))
        model.add(BatchNormalization())
        model.add(Dropout(dr + 0.1))

        model.add(Dense(128, activation="relu", kernel_regularizer=l2(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(dr))

        model.add(Dense(64, activation="relu", kernel_regularizer=l2(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(max(0.1, dr - 0.1)))

        if self.task == "classification":
            if n_classes == 2:
                model.add(Dense(1, activation="sigmoid"))
                model.compile(optimizer="adam", loss="binary_crossentropy",
                              metrics=["accuracy"])
            else:
                model.add(Dense(n_classes, activation="softmax"))
                model.compile(optimizer="adam",
                              loss="sparse_categorical_crossentropy",
                              metrics=["accuracy"])
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return model

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if self.task == "classification":
            y_enc = self._le.fit_transform(y)
            n_classes = len(self._le.classes_)
            self._model = self._build(X.shape[1], n_classes)
            self._history = self._model.fit(
                X, y_enc, epochs=self.epochs,
                batch_size=self.batch_size, verbose=0)
        else:
            y_f = np.asarray(y, dtype=float)
            self._model = self._build(X.shape[1], 1)
            self._history = self._model.fit(
                X, y_f, epochs=self.epochs,
                batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        raw = self._model.predict(X, verbose=0)
        if self.task == "classification":
            n_classes = len(self._le.classes_)
            y_enc = (raw[:, 0] >= 0.5).astype(int) if n_classes == 2 \
                     else np.argmax(raw, axis=1)
            return self._le.inverse_transform(y_enc)
        return raw[:, 0]

    def predict_proba(self, X):
        if self.task != "classification":
            return None
        X = np.asarray(X, dtype=float)
        raw = self._model.predict(X, verbose=0)
        n_classes = len(self._le.classes_)
        return np.column_stack([1 - raw[:, 0], raw[:, 0]]) \
               if n_classes == 2 else raw

    def get_metrics(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        if self.task == "classification":
            return _classification_metrics(y_test, y_pred, self.predict_proba(X_test))
        return _regression_metrics(y_test, y_pred)


# ─────────────────────────────────────────────────────────────────
# 4. CNN 1D (Convolucional sobre datos tabulares)
# ─────────────────────────────────────────────────────────────────

class CNN1DModel:
    """
    Red Neuronal Convolucional 1D aplicada a datos tabulares.

    Los features se tratan como una secuencia temporal de longitud n_features.
    Architecture: Input(n,1) → Conv1D(64,3,ReLU) → Conv1D(32,3,ReLU)
                             → GlobalAvgPool1D → Dense(64,ReLU) → Output
    """

    NAME = "CNN 1D (Keras)"

    def __init__(self, task: str = "classification", epochs: int = 50,
                 batch_size: int = 32, filters: int = 64,
                 kernel_size: int = 3, random_state: int = 42):
        self.task        = task
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.filters     = filters
        self.kernel_size = kernel_size
        self.random_state = random_state
        self._model      = None
        self._le         = LabelEncoder()
        self._history    = None

    def _build(self, n_features: int, n_classes: int):
        import tensorflow as tf
        tf.random.set_seed(self.random_state)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Conv1D, GlobalAveragePooling1D,
                                              Dense, Dropout)

        ks = min(self.kernel_size, n_features - 1) if n_features > 1 else 1

        model = Sequential(name="CNN1D")
        model.add(Conv1D(self.filters, ks, activation="relu",
                         input_shape=(n_features, 1), padding="same"))
        model.add(Conv1D(self.filters // 2, ks, activation="relu", padding="same"))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        if self.task == "classification":
            if n_classes == 2:
                model.add(Dense(1, activation="sigmoid"))
                model.compile(optimizer="adam", loss="binary_crossentropy",
                              metrics=["accuracy"])
            else:
                model.add(Dense(n_classes, activation="softmax"))
                model.compile(optimizer="adam",
                              loss="sparse_categorical_crossentropy",
                              metrics=["accuracy"])
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return model

    def fit(self, X, y):
        X_3d = np.asarray(X, dtype=float).reshape(len(X), X.shape[1], 1)
        if self.task == "classification":
            y_enc = self._le.fit_transform(y)
            n_classes = len(self._le.classes_)
            self._model = self._build(X.shape[1], n_classes)
            self._history = self._model.fit(
                X_3d, y_enc, epochs=self.epochs,
                batch_size=self.batch_size, verbose=0)
        else:
            y_f = np.asarray(y, dtype=float)
            self._model = self._build(X.shape[1], 1)
            self._history = self._model.fit(
                X_3d, y_f, epochs=self.epochs,
                batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        X_3d = np.asarray(X, dtype=float).reshape(len(X), X.shape[1], 1)
        raw  = self._model.predict(X_3d, verbose=0)
        if self.task == "classification":
            n_classes = len(self._le.classes_)
            y_enc = (raw[:, 0] >= 0.5).astype(int) if n_classes == 2 \
                     else np.argmax(raw, axis=1)
            return self._le.inverse_transform(y_enc)
        return raw[:, 0]

    def predict_proba(self, X):
        if self.task != "classification":
            return None
        X_3d = np.asarray(X, dtype=float).reshape(len(X), X.shape[1], 1)
        raw  = self._model.predict(X_3d, verbose=0)
        n_classes = len(self._le.classes_)
        return np.column_stack([1 - raw[:, 0], raw[:, 0]]) \
               if n_classes == 2 else raw

    def get_metrics(self, X_test, y_test) -> dict:
        y_pred = self.predict(X_test)
        if self.task == "classification":
            return _classification_metrics(y_test, y_pred, self.predict_proba(X_test))
        return _regression_metrics(y_test, y_pred)


# ─────────────────────────────────────────────────────────────────
# 5. AUTOENCODER KERAS (Detección de Anomalías)
# ─────────────────────────────────────────────────────────────────

class AutoencoderModel:
    """
    Autoencoder para reducción de dimensionalidad y detección de anomalías.

    Architecture:
        Encoder: Input → Dense(64,ReLU) → Dense(32,ReLU) → Dense(latent,ReLU)
        Decoder: Latent → Dense(32,ReLU) → Dense(64,ReLU) → Dense(n_features)

    En clasificación: usa el error de reconstrucción como score de anomalía.
    En regresión:     retorna el error de reconstrucción por muestra.

    Nota: Es un modelo no-supervisado; y se usa para detectar
          qué productos tienen características atípicas (precio anómalo).
    """

    NAME = "Autoencoder (Keras)"

    def __init__(self, latent_dim: int = 8, epochs: int = 80,
                 batch_size: int = 32, threshold_pct: float = 90.0,
                 random_state: int = 42):
        self.latent_dim    = latent_dim
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.threshold_pct = threshold_pct   # percentil para clasificar anomalía
        self.random_state  = random_state
        self._encoder      = None
        self._autoencoder  = None
        self._threshold    = None
        self._history      = None
        self._le           = LabelEncoder()
        self._task         = "anomaly"  # siempre anomaly detection

    def _build(self, n_features: int):
        import tensorflow as tf
        tf.random.set_seed(self.random_state)
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Dense, Input

        inp = Input(shape=(n_features,), name="input")
        enc = Dense(64, activation="relu", name="enc_1")(inp)
        enc = Dense(32, activation="relu", name="enc_2")(enc)
        lat = Dense(self.latent_dim, activation="relu", name="latent")(enc)

        dec = Dense(32, activation="relu", name="dec_1")(lat)
        dec = Dense(64, activation="relu", name="dec_2")(dec)
        out = Dense(n_features, activation="linear", name="output")(dec)

        autoencoder = Model(inputs=inp, outputs=out, name="Autoencoder")
        autoencoder.compile(optimizer="adam", loss="mse")

        encoder = Model(inputs=inp, outputs=lat, name="Encoder")

        return autoencoder, encoder

    def fit(self, X, y=None):
        """Entrena el autoencoder en modo no supervisado (ignora y)."""
        X = np.asarray(X, dtype=float)
        self._autoencoder, self._encoder = self._build(X.shape[1])
        self._history = self._autoencoder.fit(
            X, X,                        # reconstruye la entrada
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
        )
        # Calcular threshold de anomalía basado en datos de entrenamiento
        recon = self._autoencoder.predict(X, verbose=0)
        errors = np.mean((X - recon) ** 2, axis=1)
        self._threshold = float(np.percentile(errors, self.threshold_pct))
        return self

    def reconstruction_error(self, X) -> np.ndarray:
        """Error cuadrático medio de reconstrucción por muestra."""
        X = np.asarray(X, dtype=float)
        recon = self._autoencoder.predict(X, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)

    def predict(self, X) -> np.ndarray:
        """
        Retorna 1 (anomalía) o 0 (normal) según el threshold de reconstrucción.
        """
        errors = self.reconstruction_error(X)
        return (errors > self._threshold).astype(int)

    def encode(self, X) -> np.ndarray:
        """Retorna la representación latente (embedding) de los datos."""
        X = np.asarray(X, dtype=float)
        return self._encoder.predict(X, verbose=0)

    def get_metrics(self, X_test, y_test=None) -> dict:
        """
        Métricas del autoencoder:
        - error_medio_reconstruccion
        - anomalias_detectadas
        - threshold
        Si se provee y_test (etiquetas de anomalía 0/1), calcula accuracy.
        """
        errors = self.reconstruction_error(X_test)
        preds  = (errors > self._threshold).astype(int)
        met = {
            "Error Reconstrucción (medio)": round(float(errors.mean()), 6),
            "Error Reconstrucción (max)":   round(float(errors.max()), 6),
            "Threshold":                    round(self._threshold, 6),
            "Anomalías detectadas":         int(preds.sum()),
            "% Anomalías":                  round(float(preds.mean() * 100), 2),
        }
        if y_test is not None:
            try:
                met["Accuracy"] = round(float(accuracy_score(y_test, preds)), 4)
                met["F1-Score"] = round(float(
                    f1_score(y_test, preds, zero_division=0)), 4)
            except Exception:
                pass
        return met

    @property
    def history(self):
        return self._history.history if self._history else {}


# ─────────────────────────────────────────────────────────────────
# FUNCIÓN DE BENCHMARKING INTEGRADO
# ─────────────────────────────────────────────────────────────────

def benchmark_neural_networks(
    X_train, X_test, y_train, y_test,
    task: str = "classification",
    epochs: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Entrena y evalúa los 5 tipos de redes neuronales sobre los mismos datos.

    Parameters
    ----------
    X_train, X_test : arrays de features (ya escalados recomendado)
    y_train, y_test : arrays de target
    task            : 'classification' | 'regression'
    epochs          : épocas para modelos Keras
    random_state    : semilla

    Returns
    -------
    pd.DataFrame con métricas por arquitectura
    """
    keras_ok = _check_keras()

    X_train = np.asarray(X_train, dtype=float)
    X_test  = np.asarray(X_test,  dtype=float)

    models_to_run = [
        MLPSklearnModel(task=task, random_state=random_state),
    ]

    if keras_ok:
        models_to_run += [
            FeedforwardKerasModel(task=task, epochs=epochs, random_state=random_state),
            DeepDropoutModel(task=task, epochs=epochs, random_state=random_state),
            CNN1DModel(task=task, epochs=epochs, random_state=random_state),
            AutoencoderModel(epochs=epochs, random_state=random_state),
        ]
    else:
        # Fallback: solo sklearn con configuraciones distintas
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        models_to_run += [
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

    for model in models_to_run:
        name = model.NAME
        try:
            # Autoencoder: entrenamiento no supervisado
            if isinstance(model, AutoencoderModel):
                model.fit(X_train)
                met = model.get_metrics(X_test)
                met["Modelo"] = name
                met["Tipo"]   = "No Supervisado"
                results.append(met)
                continue

            model.fit(X_train, y_train)
            met = model.get_metrics(X_test, y_test)
            met["Modelo"] = name
            met["Tipo"]   = "Supervisado"
            results.append(met)

        except Exception as e:
            results.append({
                "Modelo": name,
                "Tipo":   "Error",
                "_error": str(e),
            })

    df = pd.DataFrame(results)

    # Reordenar columnas
    first_cols = ["Modelo", "Tipo"]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    return df


# ─────────────────────────────────────────────────────────────────
# RESUMEN DE ARQUITECTURAS
# ─────────────────────────────────────────────────────────────────

ARCHITECTURES_INFO = {
    "MLP (sklearn)": {
        "Tipo":         "Perceptrón Multicapa",
        "Framework":    "scikit-learn",
        "Capas":        "Input → Dense(64) → Dense(32) → Dense(16) → Output",
        "Activación":   "ReLU (ocultas) · Softmax/Lineal (salida)",
        "Optimizer":    "Adam",
        "Regularización": "Ninguna",
        "Uso":          "Referencia base, rápido y robusto",
    },
    "Feedforward (Keras)": {
        "Tipo":         "Red Densa Básica",
        "Framework":    "TensorFlow / Keras",
        "Capas":        "Input → Dense(128,ReLU) → Dense(64,ReLU) → Dense(32,ReLU) → Output",
        "Activación":   "ReLU (ocultas) · Sigmoid/Softmax/Lineal (salida)",
        "Optimizer":    "Adam",
        "Regularización": "Ninguna",
        "Uso":          "Línea base con Keras, más capas que MLP sklearn",
    },
    "Deep + Dropout (Keras)": {
        "Tipo":         "Red Profunda Regularizada",
        "Framework":    "TensorFlow / Keras",
        "Capas":        "Dense(256) → BN → Drop → Dense(128) → BN → Drop → Dense(64) → BN → Drop → Output",
        "Activación":   "ReLU + BatchNormalization",
        "Optimizer":    "Adam",
        "Regularización": "L2 + Dropout(0.4/0.3/0.2) + BatchNorm",
        "Uso":          "Previene overfitting en datasets pequeños",
    },
    "CNN 1D (Keras)": {
        "Tipo":         "Red Convolucional 1D",
        "Framework":    "TensorFlow / Keras",
        "Capas":        "Conv1D(64,k=3) → Conv1D(32,k=3) → GlobalAvgPool → Dense(64) → Output",
        "Activación":   "ReLU",
        "Optimizer":    "Adam",
        "Regularización": "Dropout(0.2)",
        "Uso":          "Captura patrones locales en la secuencia de features",
    },
    "Autoencoder (Keras)": {
        "Tipo":         "Autoencoder (No Supervisado)",
        "Framework":    "TensorFlow / Keras",
        "Capas":        "Dense(64)→Dense(32)→Dense(latent=8) | Dense(32)→Dense(64)→Output",
        "Activación":   "ReLU (encoder/decoder) · Lineal (reconstrucción)",
        "Optimizer":    "Adam",
        "Regularización": "Cuello de botella latente",
        "Uso":          "Detección de anomalías / reducción dimensional",
    },
}