# ================================================================
# ЧИСТИЙ ПРЕПРОЦЕС БЕЗ DATA LEAKAGE
# ================================================================

import os
from typing import Dict, Any, List, Optional
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, RocCurveDisplay,
    roc_auc_score, f1_score
)
import matplotlib.pyplot as plt


# -------------------- ДОПОМОЖНІ ФУНКЦІЇ --------------------

def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Додаємо всі похідні фічі ДО спліту."""
    df = df.copy()

    # Drop технічних колонок
    df = df.drop(['id', 'CustomerId', 'Surname'], axis=1, errors='ignore')

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
        save_dir="models",
        scaler_numeric=True,
        test_size=0.1,
) -> Dict[str, Any]:

    os.makedirs(save_dir, exist_ok=True)

    # 1) ДОДАЄМО ФІЧІ ДО СПЛІТУ
    df = _add_features(raw_df)

    target_col = "Exited"
    numeric_cols = ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance']
    categorical_cols = ['Geography', 'Gender', 'ProductGroup']

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

    # 3) ВИДІЛЯЄМО Y
    y_train, y_val, y_test = (
        train_df[target_col].copy(),
        val_df[target_col].copy(),
        test_df[target_col].copy(),
    )

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
    train_encoded, val_encoded, encoder = _encode(train_df, val_df, categorical_cols)
    _, test_encoded, _ = _encode(train_df, test_df, categorical_cols)

    # 6) Об’єднання фінальних фіч
    X_train = pd.concat([train_df.drop(columns=categorical_cols), train_encoded], axis=1)
    X_val = pd.concat([val_df.drop(columns=categorical_cols), val_encoded], axis=1)
    X_test = pd.concat([test_df.drop(columns=categorical_cols), test_encoded], axis=1)

    input_cols = list(X_train.columns)

    # 7) Збереження preprocessors
    if scaler:
        joblib.dump(scaler, f"{save_dir}/scaler.joblib")
    joblib.dump(encoder, f"{save_dir}/encoder.joblib")

    print("✅ Препроцес завершено без leakage")

    return {
        'train_X': X_train,
        'train_y': y_train,
        'val_X': X_val,
        'val_y': y_val,
        'test_X': X_test,
        'test_y': y_test,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder,
    }


# -------------------- ТРАНСФОРМАЦІЯ НОВИХ ДАНИХ --------------------

def transform_new_data(new_df: pd.DataFrame,
                       preprocessors: Dict[str, Optional[Any]]):
    df = _add_features(new_df)

    numeric_cols = ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance']
    categorical_cols = ['Geography', 'Gender', 'ProductGroup']

    scaler = preprocessors.get("scaler")
    encoder: OneHotEncoder = preprocessors.get("encoder")

    if scaler:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    df = pd.concat([df.drop(columns=categorical_cols), encoded], axis=1)
    return df
