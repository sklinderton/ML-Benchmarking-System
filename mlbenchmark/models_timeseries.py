"""
models_timeseries.py - Modelos de Series de Tiempo: ARIMA, Holt-Winters, LSTM
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")


class HoltWintersModel:
    """Wrapper para Holt-Winters (Triple Exponential Smoothing)."""

    def __init__(self, seasonal_periods=12, trend="add", seasonal="add",
                 initialization_method="estimated"):
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.initialization_method = initialization_method
        self.model_ = None
        self.fitted_ = None

    def fit(self, train):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.model_ = ExponentialSmoothing(
            train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method=self.initialization_method,
        )
        self.fitted_ = self.model_.fit(optimized=True, disp=False)
        return self

    def predict(self, steps):
        return self.fitted_.forecast(steps)


class HoltWintersCalibrated:
    """Holt-Winters con búsqueda de configuración óptima (add vs mul)."""

    def __init__(self, seasonal_periods=12):
        self.seasonal_periods = seasonal_periods
        self.best_model_ = None
        self.best_config_ = None

    def fit(self, train):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from sklearn.metrics import mean_squared_error

        configs = [
            {"trend": "add", "seasonal": "add"},
            {"trend": "add", "seasonal": "mul"},
            {"trend": "mul", "seasonal": "add"},
            {"trend": "mul", "seasonal": "mul"},
        ]

        best_mse = np.inf

        # Usa últimos seasonal_periods como validación interna
        split = len(train) - self.seasonal_periods
        t_tr, t_val = train[:split], train[split:]

        for cfg in configs:
            try:
                m = ExponentialSmoothing(
                    t_tr,
                    trend=cfg["trend"],
                    seasonal=cfg["seasonal"],
                    seasonal_periods=self.seasonal_periods,
                    initialization_method="estimated",
                )
                fit = m.fit(optimized=True, disp=False)
                pred = fit.forecast(len(t_val))
                mse = mean_squared_error(t_val, pred)
                if mse < best_mse:
                    best_mse = mse
                    self.best_config_ = cfg
            except Exception:
                continue

        # Reentrenar con toda la serie
        if self.best_config_:
            m = ExponentialSmoothing(
                train,
                trend=self.best_config_["trend"],
                seasonal=self.best_config_["seasonal"],
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            self.best_model_ = m.fit(optimized=True, disp=False)

        return self

    def predict(self, steps):
        return self.best_model_.forecast(steps)


class ARIMAModel:
    """Wrapper para ARIMA con parámetros fijos."""

    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.fitted_ = None

    def fit(self, train):
        from statsmodels.tsa.arima.model import ARIMA
        m = ARIMA(train, order=self.order)
        self.fitted_ = m.fit()
        return self

    def predict(self, steps):
        forecast = self.fitted_.forecast(steps=steps)
        return forecast


class ARIMACalibrated:
    """ARIMA con búsqueda automática de orden óptimo (p,d,q)."""

    def __init__(self, max_p=3, max_d=2, max_q=3):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.best_order_ = None
        self.fitted_ = None

    def fit(self, train):
        from statsmodels.tsa.arima.model import ARIMA
        from itertools import product

        best_aic = np.inf
        best_order = (1, 1, 1)

        for p, d, q in product(range(self.max_p + 1),
                                range(self.max_d + 1),
                                range(self.max_q + 1)):
            try:
                m = ARIMA(train, order=(p, d, q))
                fit = m.fit()
                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_order = (p, d, q)
            except Exception:
                continue

        self.best_order_ = best_order
        m = ARIMA(train, order=best_order)
        self.fitted_ = m.fit()
        return self

    def predict(self, steps):
        return self.fitted_.forecast(steps=steps)


class LSTMModel:
    """Red LSTM para predicción de series de tiempo."""

    def __init__(self, units=50, layers=2, epochs=50, batch_size=16,
                 window_size=12, dropout=0.2):
        self.units = units
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.dropout = dropout
        self.model_ = None
        self.scaler_params_ = None

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential()
        for i in range(self.layers):
            return_seq = (i < self.layers - 1)
            model.add(LSTM(self.units, return_sequences=return_seq,
                           input_shape=(self.window_size, 1) if i == 0 else None))
            model.add(Dropout(self.dropout))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, train):
        from mlbenchmark.preprocessing import normalize_timeseries, create_sequences

        # Normalizar
        min_val = np.min(train)
        max_val = np.max(train)
        denom = max_val - min_val if max_val != min_val else 1.0
        train_norm = (train - min_val) / denom
        self.scaler_params_ = (min_val, max_val)

        X, y = create_sequences(train_norm, self.window_size)
        self.model_ = self._build_model()
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                        verbose=0)
        self._last_window = train_norm[-self.window_size:]
        return self

    def predict(self, steps):
        min_val, max_val = self.scaler_params_
        denom = max_val - min_val if max_val != min_val else 1.0

        preds = []
        window = list(self._last_window)

        for _ in range(steps):
            x = np.array(window[-self.window_size:]).reshape(1, self.window_size, 1)
            p = self.model_.predict(x, verbose=0)[0, 0]
            preds.append(p)
            window.append(p)

        return np.array(preds) * denom + min_val


def get_timeseries_models(seasonal_periods=12):
    """
    Retorna diccionario de modelos de series de tiempo.

    Returns:
        dict: {nombre: instancia_de_modelo}
    """
    return {
        "Holt-Winters": HoltWintersModel(seasonal_periods=seasonal_periods),
        "Holt-Winters Calibrado": HoltWintersCalibrated(seasonal_periods=seasonal_periods),
        "ARIMA(1,1,1)": ARIMAModel(order=(1, 1, 1)),
        "ARIMA Calibrado": ARIMACalibrated(max_p=2, max_d=2, max_q=2),
        "LSTM": LSTMModel(units=50, layers=2, epochs=30, window_size=min(12, seasonal_periods)),
    }
