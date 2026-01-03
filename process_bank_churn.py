import os
from typing import Dict, Any, List, Optional, Iterable
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, RocCurveDisplay,
    roc_auc_score, f1_score
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt


# -------------------- ДОПОМОЖНІ ФУНКЦІЇ --------------------

def _add_features(
    df: pd.DataFrame,
    technical_columns: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Додаємо всі похідні фічі ДО спліту.

    Parameters
    ----------
    df : pd.DataFrame
        Вхідний датафрейм
    technical_columns : Iterable[str] | None
        Список технічних колонок для видалення
    """
    df = df.copy()

    # Значення за замовчуванням
    if technical_columns is None:
        technical_columns = ['id', 'CustomerId', 'Surname']

    # Drop технічних колонок
    df = df.drop(list(technical_columns), axis=1, errors='ignore')

    # Додаємо ProductGroup
    def simplify_products(x: int) -> str:
        if x == 1:
            return '1'
        elif x == 2:
            return '2'
        else:
            return '3'

    df['ProductGroup'] = df['NumOfProducts'].apply(simplify_products)

    return df


def _scale(train_df: pd.DataFrame, other_df: pd.DataFrame,
           numeric_cols: List[str]):
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])

    train_scaled = train_df.copy()
    other_scaled = other_df.copy()

    train_scaled[numeric_cols] = scaler.transform(train_df[numeric_cols])
    other_scaled[numeric_cols] = scaler.transform(other_df[numeric_cols])

    return train_scaled, other_scaled, scaler


def _encode(train_df: pd.DataFrame, other_df: pd.DataFrame,
            cat_cols: List[str]):
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(train_df[cat_cols])

    train_encoded = pd.DataFrame(
        encoder.transform(train_df[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=train_df.index
    )

    other_encoded = pd.DataFrame(
        encoder.transform(other_df[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=other_df.index
    )

    return train_encoded, other_encoded, encoder


# -------------------- ОСНОВНА ФУНКЦІЯ --------------------

def preprocess_data(
    raw_df: pd.DataFrame,
    target_col: str,
    numeric_cols: Iterable[str],
    categorical_cols: Iterable[str],
    technical_columns: Optional[Iterable[str]] = None,
    save_dir: str = "models",
    scaler_numeric: bool = True,
    test_size: float = 0.1,
) -> Dict[str, Any]:

    os.makedirs(save_dir, exist_ok=True)

    # 1) ДОДАЄМО ФІЧІ ДО СПЛІТУ
    df = _add_features(raw_df, technical_columns)

    # 2) СПЛІТ
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[target_col]
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.2,
        random_state=42,
        stratify=train_val_df[target_col]
    )

    # 3) ВИДІЛЯЄМО y
    y_train = train_df[target_col].copy()
    y_val = val_df[target_col].copy()
    y_test = test_df[target_col].copy()

    # Видаляємо таргет з X
    train_df = train_df.drop(columns=[target_col])
    val_df = val_df.drop(columns=[target_col])
    test_df = test_df.drop(columns=[target_col])

    # 4) Масштабування (без leakage)
    scaler = None
    if scaler_numeric:
        train_df, val_df, scaler = _scale(train_df, val_df, numeric_cols)
        _, test_df, _ = _scale(train_df, test_df, numeric_cols)
    else:
        print("🚫 Масштабування числових ознак вимкнено.")

    # 5) One-Hot Encoding (без leakage)
    train_encoded, val_encoded, encoder = _encode(
        train_df, val_df, categorical_cols
    )
    _, test_encoded, _ = _encode(train_df, test_df, categorical_cols)

    # 6) Фінальні X
    X_train = pd.concat(
        [train_df.drop(columns=categorical_cols), train_encoded], axis=1
    )
    X_val = pd.concat(
        [val_df.drop(columns=categorical_cols), val_encoded], axis=1
    )
    X_test = pd.concat(
        [test_df.drop(columns=categorical_cols), test_encoded], axis=1
    )

    input_cols = list(X_train.columns)

    # 7) Збереження preprocessors
    if scaler:
        joblib.dump(scaler, f"{save_dir}/scaler.joblib")
    joblib.dump(encoder, f"{save_dir}/encoder.joblib")

    print("✅ Препроцес завершено")

    return {
        "train_X": X_train,
        "train_y": y_train,
        "val_X": X_val,
        "val_y": y_val,
        "test_X": X_test,
        "test_y": y_test,
        "input_cols": input_cols,
        "scaler": scaler,
        "encoder": encoder,
        "numeric_cols": list(numeric_cols),
        "categorical_cols": list(categorical_cols),
        "technical_columns": list(technical_columns) if technical_columns else None,
        "target_col": target_col,
    }


# -------------------- ТРАНСФОРМАЦІЯ НОВИХ ДАНИХ --------------------

def transform_new_data(
    new_df: pd.DataFrame,
    preprocessors: Dict[str, Any],
):
    df = _add_features(new_df, preprocessors.get("technical_columns"))

    numeric_cols = preprocessors["numeric_cols"]
    categorical_cols = preprocessors["categorical_cols"]

    scaler = preprocessors.get("scaler")
    encoder: OneHotEncoder = preprocessors["encoder"]

    # Масштабування
    if scaler:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # OHE
    encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    df = pd.concat(
        [df.drop(columns=categorical_cols), encoded],
        axis=1
    )

    # Вирівнюємо порядок колонок
    df = df[preprocessors["input_cols"]]

    return df

# -------------------- ОЦІНКА МОДЕЛІ --------------------

def evaluate_model_from_proba(y_true, y_proba, dataset_name='Dataset'):
    """
    Оцінює модель за готовими true labels та predicted probabilities.

    Args:
        y_true (array-like): Реальні значення цільової змінної (0/1).
        y_proba (array-like): Прогнозовані ймовірності класу 1.
        dataset_name (str): Назва датасету для підписів графіків.
    """

    # 1. Перетворення порогу 0.5 на класи
    y_pred = (y_proba >= 0.5).astype(int)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix: {dataset_name}')
    plt.show()

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title(f'ROC Curve: {dataset_name}')
    plt.show()

    # 4. Метрики
    auc = roc_auc_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred)

    print(f"📊 {dataset_name} — AUROC: {auc:.3f}, F1 Score (threshold=0.5): {f1:.3f}")

    #----------------------- Для Pipeline ----------------------------------------------------


    # --- 1) Кастомний трансформер: видалити технічні колонки + додати ProductGroup
class AddFeaturesDropTech(BaseEstimator, TransformerMixin):
    def __init__(self, technical_columns=None):
        self.technical_columns = technical_columns or ['id', 'CustomerId', 'Surname']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # 1) drop технічних колонок
        X = X.drop(columns=list(self.technical_columns), errors='ignore')

        # 2) додати ProductGroup (з NumOfProducts)
        def simplify_products(n):
            if n == 1:
                return "1"
            elif n == 2:
                return "2"
            else:
                return "3"

        if "NumOfProducts" in X.columns:
            X["ProductGroup"] = X["NumOfProducts"].apply(simplify_products)

        return X

def build_pipeline(numeric_cols, categorical_cols, technical_columns=None, random_state=42):
    # Препроцесинг колонок (робиться всередині pipeline)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), list(numeric_cols)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline(steps=[
        ("features", AddFeaturesDropTech(technical_columns=technical_columns)),
        ("preprocess", preprocessor),
        ("model", DecisionTreeClassifier(random_state=random_state)),
    ])

    return pipe