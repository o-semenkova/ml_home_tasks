import os
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib


# -------------------- ВНУТРІШНІ ПІДФУНКЦІЇ --------------------

def _drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Видаляє технічні або ідентифікаційні колонки, не потрібні для навчання моделі.

    Args:
        df (pd.DataFrame): Вхідний датафрейм.

    Returns:
        pd.DataFrame: Датафрейм без технічних колонок.
    """
    return df.drop(['id', 'CustomerId', 'Surname'], axis=1, errors='ignore')


def _add_product_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Створює нову категоріальну ознаку 'ProductGroup' на основі 'NumOfProducts'.

    Args:
        df (pd.DataFrame): Вхідний датафрейм із колонкою NumOfProducts.

    Returns:
        pd.DataFrame: Копія датафрейму з доданою колонкою 'ProductGroup'.
    """
    def simplify_products(x: int) -> str:
        if x == 1:
            return '1'
        elif x == 2:
            return '2'
        else:
            return '3'

    df['ProductGroup'] = df['NumOfProducts'].apply(simplify_products)
    return df


def _scale_numeric_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Масштабує числові ознаки за допомогою MinMaxScaler.

    Args:
        train_df (pd.DataFrame): Тренувальний набір.
        val_df (pd.DataFrame): Валідаційний набір.
        numeric_cols (List[str]): Список назв числових колонок.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
            Масштабовані train і val датафрейми, а також scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()

    train_scaled[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_scaled[numeric_cols] = scaler.transform(val_df[numeric_cols])

    return train_scaled, val_scaled, scaler


def _encode_categorical_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Виконує One-Hot encoding для категоріальних ознак.

    Args:
        train_df (pd.DataFrame): Тренувальний набір.
        val_df (pd.DataFrame): Валідаційний набір.
        categorical_cols (List[str]): Список категоріальних колонок.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
            Закодовані train і val датафрейми, encoder і список назв нових колонок.
    """
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(train_df[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)

    train_encoded = pd.DataFrame(
        encoder.transform(train_df[categorical_cols]),
        columns=encoded_cols,
        index=train_df.index
    )

    val_encoded = pd.DataFrame(
        encoder.transform(val_df[categorical_cols]),
        columns=encoded_cols,
        index=val_df.index
    )

    return train_encoded, val_encoded, encoder, list(encoded_cols)


# -------------------- ПУБЛІЧНІ ФУНКЦІЇ --------------------

def preprocess_data(
    raw_df: pd.DataFrame,
    save_dir: str = "models",
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Optional[MinMaxScaler], OneHotEncoder]:
    """
    Повна попередня обробка даних для задачі класифікації відтоку клієнтів банку.

    Args:
        raw_df (pd.DataFrame): Сирий датафрейм.
        save_dir (str): Тека для збереження препроцесорів.
        scaler_numeric (bool): Чи масштабувати числові ознаки (для дерев False).

    Returns:
        Tuple: X_train, y_train, X_val, y_val, input_cols, scaler, encoder
    """
    os.makedirs(save_dir, exist_ok=True)

    df = _drop_unused_columns(raw_df)
    target_col = 'Exited'

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target_col]
    )

    numeric_cols = ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance']
    categorical_cols = ['Geography', 'Gender', 'ProductGroup']

    train_df = _add_product_group(train_df)
    val_df = _add_product_group(val_df)

    scaler: Optional[MinMaxScaler] = None
    if scaler_numeric:
        train_df, val_df, scaler = _scale_numeric_features(train_df, val_df, numeric_cols)
        print("📏 Масштабування числових ознак увімкнено.")
    else:
        print("🚫 Масштабування числових ознак вимкнено.")

    train_encoded, val_encoded, encoder, encoded_cols = _encode_categorical_features(
        train_df, val_df, categorical_cols
    )

    X_train = pd.concat([train_df.drop(columns=categorical_cols), train_encoded], axis=1)
    X_val = pd.concat([val_df.drop(columns=categorical_cols), val_encoded], axis=1)

    y_train = train_df[target_col]
    y_val = val_df[target_col]

    input_cols = X_train.columns.tolist()

    # Збереження
    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(save_dir, "encoder.joblib"))
    print(f"✅ Збережено encoder (і scaler, якщо використовувався) у '{save_dir}'")

    return X_train, y_train, X_val, y_val, input_cols, scaler, encoder


def load_preprocessors(save_dir: str = "models") -> Tuple[Optional[MinMaxScaler], Optional[OneHotEncoder]]:
    """
    Завантажує збережені scaler та encoder із директорії `save_dir`.

    Args:
        save_dir (str): Шлях до директорії з joblib-файлами.

    Returns:
        Tuple[Optional[MinMaxScaler], Optional[OneHotEncoder]]: Завантажені об’єкти або None.
    """
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    encoder_path = os.path.join(save_dir, "encoder.joblib")

    scaler = encoder = None

    if os.path.exists(encoder_path):
        encoder = joblib.load(encoder_path)
        print("✅ Завантажено encoder")
    else:
        print(f"⚠️  Не знайдено файл: {encoder_path}")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("✅ Завантажено scaler")
    else:
        print(f"ℹ️  Масштабування, ймовірно, не застосовувалось ({scaler_path} відсутній)")

    return scaler, encoder


def transform_new_data(
    new_df: pd.DataFrame,
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Обробляє нові дані тим самим способом, що й тренувальні.

    Args:
        new_df (pd.DataFrame): Нові сирі дані (без колонки 'Exited').
        scaler (Optional[MinMaxScaler]): Масштабувальник (може бути None).
        encoder (OneHotEncoder): Кодер для категоріальних змінних.

    Returns:
        pd.DataFrame: Оброблений датафрейм, готовий для моделі.
    """
    df = new_df.copy()
    df = _drop_unused_columns(df)
    df = _add_product_group(df)

    numeric_cols = ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance']
    categorical_cols = ['Geography', 'Gender', 'ProductGroup']

    if scaler:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_array = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    df_final = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    return df_final
