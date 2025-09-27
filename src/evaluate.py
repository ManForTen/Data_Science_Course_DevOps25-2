import os
import sys
import joblib
import json
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split

if __name__ == "__main__":
    models_dir = sys.argv[1]    
    data_csv = sys.argv[2]      
    metrics_path = sys.argv[3]  

    # Загружаем данные
    df = pd.read_csv(data_csv)
    X = df.drop("y", axis=1)
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    metrics = {}

    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        f1_train = f1_score(y_train, model.predict(X_train), average="macro")
        f1_test = f1_score(y_test, y_pred, average="macro")
        cvs = cross_val_score(model, X_train, y_train, cv=5).tolist()

        metrics[model_name] = {
            "classification_report": report,
            "f1_train": f1_train,
            "f1_test": f1_test,
            "cross_val_score": cvs
        }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"Метрики сохранены в {metrics_path}")
