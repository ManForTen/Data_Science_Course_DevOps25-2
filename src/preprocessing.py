import os
import sys
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import visualisation


def encode_categorical(data, target='y'):
    """One-Hot-Encoding для категориальных признаков"""
    ohe = OneHotEncoder()
    for col in data.columns:
        if data[col].dtype == 'object' and col != target:
            data[col] = data[col].fillna('null_' + col)
            transformed = ohe.fit_transform(data[[col]])
            data[ohe.categories_[0]] = transformed.toarray()
            data = data.drop(col, axis=1)
    return data


def fill_missing(data, method="constant"):
    """
    Заполнение пропусков:
    - constant → -1000
    - median → медианой
    """
    if method == "constant":
        data = data.fillna(-1000)
    elif method == "median":
        for col in data.columns:
            if data[col].dtype != 'object':
                data[col] = data[col].fillna(data[col].median())
    return data


def map_target(data, target='y'):
    """yes/no → 1/0"""
    class_dictionary = {"yes": 1, "no": 0}
    data[target] = data[target].map(class_dictionary)
    return data

if __name__ == "__main__":
    # Аргументы из командной строки: входной и выходной файл
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Загружаем датасет
    data = pd.read_csv(input_path, sep=";")

    plots_dir = os.path.join("data", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    visualisation.plot_missing(data, save_path=os.path.join(plots_dir, "missing.png"))
    visualisation.plot_unique_counts(data, save_path=os.path.join(plots_dir, "unique_counts.png"))
    visualisation.plot_correlation(data, save_path=os.path.join(plots_dir, "correlation.png"))
    visualisation.plot_numeric_distributions(data, save_dir=plots_dir)
    visualisation.plot_categorical_distributions(data, save_dir=plots_dir)

    # Применяем предобработку
    data = fill_missing(data, method="median")
    data = encode_categorical(data, target="y")
    data = map_target(data, target="y")

    # Сохраняем результат
    data.to_csv(output_path, index=False)
    print(f"Данные сохранены: {output_path}")
