"""
models_regression.py - Modelos de regresión disponibles
"""

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


def get_regression_models(random_state=42):
    """
    Retorna diccionario de modelos de regresión listos para usar.

    Returns:
        dict: {nombre: instancia_de_modelo}
    """
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0, max_iter=2000),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "SVR": SVR(kernel="rbf"),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=random_state
        ),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            random_state=random_state,
            verbosity=0,
        )
    except ImportError:
        pass

    return models


def train_regression_model(model, X_train, y_train):
    """
    Entrena un modelo de regresión.

    Returns:
        model entrenado
    """
    model.fit(X_train, y_train)
    return model
