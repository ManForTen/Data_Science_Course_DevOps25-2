import pandas as pd
from preprocessing import fill_missing, map_target


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
