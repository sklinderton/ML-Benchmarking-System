"""
models_classification.py - Modelos de clasificación disponibles
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def get_classification_models(random_state=42):
    """
    Retorna diccionario de modelos de clasificación listos para usar.

    Returns:
        dict: {nombre: instancia_de_modelo}
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=random_state
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state
        ),
    }

    # Agregar XGBoost si está disponible
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
    except ImportError:
        pass

    return models


def train_classification_model(model, X_train, y_train):
    """
    Entrena un modelo de clasificación.

    Returns:
        model entrenado
    """
    model.fit(X_train, y_train)
    return model


def predict_classification(model, X_test, threshold=0.5):
    """
    Genera predicciones de clase y probabilidades.

    Returns:
        y_pred, y_prob (o None si no tiene predict_proba)
    """
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    return y_pred, y_prob
