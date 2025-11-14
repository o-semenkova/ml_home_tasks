import os
from typing import Optional, List, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib


# -------------------- ВНУТРІШНІ ПІДФУНКЦІЇ --------------------

def _drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Видаляє технічні або ідентифікаційні колонки, не потрібні для навчання.

    Args:
        df (pd.DataFrame): Вхідний датафрейм.

    Returns:
        pd.DataFrame: Датафрейм без непотрібних колонок.
    """
    return df.drop(['id', 'CustomerId', 'Surname'], axis=1, errors='ignore')


def _add_product_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Створює нову категоріальну ознаку 'ProductGroup' на основі 'NumOfProducts'.

    Args:
        df (pd.DataFrame): Вхідний датафрейм.

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
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Масштабує числові ознаки за допомогою MinMaxScaler.

    Args:
        train_df (pd.DataFrame): Тренувальний набір.
        val_df (pd.DataFrame): Валідаційний або тестовий набір.
        numeric_cols (List[str]): Список числових колонок.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: 
        Масштабовані датафрейми та об'єкт scaler.
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
) -> tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Виконує One-Hot encoding для категоріальних ознак.

    Args:
        train_df (pd.DataFrame): Тренувальний набір.
        val_df (pd.DataFrame): Валідаційний набір.
        categorical_cols (List[str]): Список категоріальних колонок.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, OneHotEncoder, List[str]]:
        Закодовані датафрейми, encoder і список назв нових колонок.
    """
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoder.fit(train_df[categorical_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    train_encoded = pd.DataFrame(
        encoder.transform(train_df[categorical_cols]),
        columns=encoded_cols, index=train_df.index
    )
    val_encoded = pd.DataFrame(
        encoder.transform(val_df[categorical_cols]),
        columns=encoded_cols, index=val_df.index
    )
    return train_encoded, val_encoded, encoder, list(encoded_cols)


# -------------------- ПУБЛІЧНІ ФУНКЦІЇ --------------------

def preprocess_data(
    raw_df: pd.DataFrame,
    save_dir: str = "models",
    scaler_numeric: bool = True,
    test_size: float = 0.1
) -> Dict[str, Any]:
    """
    Повна попередня обробка даних для задачі класифікації відтоку клієнтів банку.

    Args:
        raw_df (pd.DataFrame): Сирий датафрейм.
        save_dir (str): Тека для збереження препроцесорів.
        scaler_numeric (bool): Чи масштабувати числові ознаки (для дерев False).
        test_size (float): Частка даних для тестового набору.

    Returns:
        dict[str, Any]: 
            Словник із train/val/test наборами, препроцесорами та списком ознак:
            {
                'train_X', 'train_y', 'val_X', 'val_y', 
                'test_X', 'test_y', 'input_cols', 'scaler', 'encoder'
            }
    """
    os.makedirs(save_dir, exist_ok=True)
    df = _drop_unused_columns(raw_df)
    target_col = 'Exited'

    # train/val/test split
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df[target_col])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df[target_col])

    numeric_cols = ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance']
    categorical_cols = ['Geography', 'Gender', 'ProductGroup']

    for subset in (train_df, val_df, test_df):
        _add_product_group(subset)

    scaler: Optional[MinMaxScaler] = None
    if scaler_numeric:
        train_df, val_df, scaler = _scale_numeric_features(train_df, val_df, numeric_cols)
        _, test_df, _ = _scale_numeric_features(train_df, test_df, numeric_cols)
        print("📏 Масштабування числових ознак увімкнено.")
    else:
        print("🚫 Масштабування числових ознак вимкнено.")

    train_encoded, val_encoded, encoder, encoded_cols = _encode_categorical_features(train_df, val_df, categorical_cols)
    test_encoded = pd.DataFrame(
        encoder.transform(test_df[categorical_cols]),
        columns=encoded_cols, index=test_df.index
    )

    X_train = pd.concat([train_df.drop(columns=categorical_cols), train_encoded], axis=1)
    X_val = pd.concat([val_df.drop(columns=categorical_cols), val_encoded], axis=1)
    X_test = pd.concat([test_df.drop(columns=categorical_cols), test_encoded], axis=1)

    y_train, y_val, y_test = train_df[target_col], val_df[target_col], test_df[target_col]
    input_cols = X_train.columns.tolist()

    # save preprocessors
    if scaler is not None:
        joblib.dump(scaler, os.path.join(save_dir, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(save_dir, "encoder.joblib"))
    print(f"✅ Збережено encoder (і scaler, якщо використовувався) у '{save_dir}'")

    return {
        'train_X': X_train,
        'train_y': y_train,
        'val_X': X_val,
        'val_y': y_val,
        'test_X': X_test,
        'test_y': y_test,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder
    }


def load_preprocessors(save_dir: str = "models") -> Dict[str, Optional[object]]:
    """
    Завантажує збережені scaler та encoder.

    Args:
        save_dir (str): Тека, де збережені joblib-файли.

    Returns:
        dict[str, Optional[object]]: {'scaler': scaler або None, 'encoder': encoder або None}
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
        print(f"ℹ️  Масштабування не застосовувалось ({scaler_path} відсутній)")

    return {'scaler': scaler, 'encoder': encoder}


def transform_new_data(
    new_df: pd.DataFrame, 
    preprocessors: Dict[str, Optional[object]]
) -> pd.DataFrame:
    """
    Обробляє нові дані тим самим способом, що й тренувальні.

    Args:
        new_df (pd.DataFrame): Нові сирі дані (без 'Exited').
        preprocessors (dict): Словник із 'scaler' і 'encoder'.

    Returns:
        pd.DataFrame: Оброблений датафрейм, готовий до моделі.
    """
    df = new_df.copy()
    df = _drop_unused_columns(df)
    df = _add_product_group(df)

    numeric_cols = ['Age', 'NumOfProducts', 'IsActiveMember', 'Balance']
    categorical_cols = ['Geography', 'Gender', 'ProductGroup']

    scaler = preprocessors.get('scaler')
    encoder = preprocessors.get('encoder')

    if scaler:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_array = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    df_final = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)
    return df_final

def evaluate_model(pipeline, X, dataset_name='Dataset'):
    # 1. Прогноз ймовірностей
    y_proba = pipeline.predict_proba(X)[:, 1]

    # 2. Прогноз класів при порозі 0.5
    y_pred = (y_proba >= 0.5).astype(int)

    # 3. Confusion Matrix
    cm = confusion_matrix(X['Exited'], y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix: {dataset_name}')
    plt.show()

    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(X['Exited'], y_proba)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title(f'ROC Curve: {dataset_name}')
    plt.show()

    # 5. Метрики
    auc = roc_auc_score(X['Exited'], y_proba)
    f1 = f1_score(X['Exited'], y_pred)

    print(f"📊 {dataset_name} — AUROC: {auc:.3f}, F1 Score (threshold=0.5): {f1:.3f}")
# -------------------- ТЕСТОВИЙ ЗАПУСК --------------------
if __name__ == "__main__":
    csv_path = "train.csv"

    if not os.path.exists(csv_path):
        print("⚠️  Файл train.csv не знайдено.")
    else:
        raw_df = pd.read_csv(csv_path)
        data = preprocess_data(raw_df, scaler_numeric=True)
        print("✅ Обробка завершена. Train shape:", data['train_X'].shape)

        new_df = raw_df.sample(3, random_state=1).drop(columns=['Exited'])
        transformed = transform_new_data(new_df, {'scaler': data['scaler'], 'encoder': data['encoder']})
        print("🔁 Приклад трансформації нових даних:")
        print(transformed.head())
