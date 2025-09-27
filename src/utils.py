import pandas as pd

def load_data(path):
    """Загрузка данных"""
    return pd.read_csv(path, delimiter=";")
