import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


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


def test_fill_missing_constant():
    """
    Тест функции fill_missing с методом 'constant'.
    Проверяет, что пропуски заменяются на значение -1000.
    """
    df = pd.DataFrame({"a": [1, None, 3]})
    result = fill_missing(df, method="constant")
    assert -1000 in result["a"].values


def test_map_target():
    """
    Тест функции map_target.
    Проверяет, что значения 'yes' и 'no' корректно маппятся в 1 и 0.
    """
    df = pd.DataFrame({"y": ["yes", "no", "yes"]})
    result = map_target(df)
    assert set(result["y"].unique()) == {0, 1}


def test_model_training():
    """
    Тест обучения модели LogisticRegression.
    Проверяет, что количество предсказаний совпадает с количеством объектов.
    """
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 0, 1, 0]})
    y = np.array([0, 1, 0, 1])
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
