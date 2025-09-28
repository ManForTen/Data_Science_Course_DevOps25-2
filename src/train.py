import os
import sys
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from yellowbrick.model_selection import LearningCurve
import visualisation
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def compare_models(models, X_train, X_test, y_train, y_test, base_dir):
    """Обучение моделей + сохранение графиков"""
    for model in models:
        try:
            model_name = type(model).__name__.lower()
            model_dir = os.path.join(base_dir, model_name)

            start_time = time.time()
            model = model.fit(X_train, y_train)
            end_time = time.time()
            print(f"Время обучения {model_name}: {end_time - start_time:.2f} секунд")

            y_pred = model.predict(X_test)

            # Создаём папку для модели
            os.makedirs(model_dir, exist_ok=True)

            # Кривая обучения
            lc = LearningCurve(model, scoring='f1_macro', train_sizes=np.linspace(0.7, 1.0, 10))
            lc.fit(X_train, y_train)
            lc.finalize()
            learning_curve_path = os.path.join(model_dir, "learning_curve.png")
            lc.ax.figure.savefig(learning_curve_path, bbox_inches="tight")

            # Матрица ошибок
            fig, ax = plt.subplots()
            sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            plt.title(f"Матрица ошибок: {model_name}")
            cm_path = os.path.join(model_dir, "confusion_matrix.png")
            visualisation.save_plot(fig, cm_path)

            # Значимость признаков
            try:
                importances = pd.DataFrame({
                    'Attribute': X_train.columns,
                    'Importance': model.feature_importances_
                })
            except:
                importances = pd.DataFrame({
                    'Attribute': X_train.columns,
                    'Importance': model.coef_[0]
                })

            importances = importances.sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots()
            ax.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
            plt.title(f'Значимость признаков: {model_name}', size=20)
            plt.xticks(rotation='vertical')
            fi_path = os.path.join(model_dir, "feature_importance.png")
            visualisation.save_plot(fig, fi_path)

            # Сохраняем модель
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            print(f"Модель {model_name} сохранена в {model_path}")

        except Exception as e:
            print(f"Ошибка при обучении {model_name}: {e}")
        
        # Отчёт
        print(f"MODEL: {str(model)}")
        print(f"ClREP:\n{classification_report(y_test, y_pred)}")
        print(f"CVS:   {cross_val_score(model, X_train, y_train, cv=5)}")
        print(f"Переобучение: {f1_score(y_train, model.predict(X_train), average='macro')} "
              f"на {f1_score(y_test, y_pred, average='macro')}")
        
        model_path = os.path.join(model_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"Модель {model_name} сохранена в {model_path}")

        print("-" * 100)


if __name__ == "__main__":
    input_csv = sys.argv[1]   
    output_dir = sys.argv[2]  

    # Загружаем обработанные данные
    df = pd.read_csv(input_csv)

    X = df.drop("y", axis=1)
    y = df["y"]

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Модели для обучения
    models = [
        GradientBoostingClassifier(random_state=42),
        RandomForestClassifier(random_state=42),
        LogisticRegression(random_state=42)
    ]

    # Обучение моделей
    compare_models(models, X_train, X_test, y_train, y_test, output_dir)
