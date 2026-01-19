from data_loader import DataLoader
from preprocessing import Preprocessor
from models import FallClassifier, RandomForestModel, SVMModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score


def main():
    DATA_PATH = "../data/processed/dataset.csv"

    # Wczytanie danych
    df = DataLoader(DATA_PATH).load()

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42    # Wykorzystanie wektorów
    )

    print("\nROZKŁAD KLAS:")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        raise RuntimeError(
            "Dataset ma tylko jedną klasę. "
            "Sprawdź build_dataset i folder labels."
        )

    # Normalizacja danych
    pre = Preprocessor()
    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)

    models = {
        "Logistic Regression": FallClassifier(),
        "Random Forest": RandomForestModel(),
        "SVM": SVMModel()
    }

    results = []

    for name, model in models.items():
        model.train(X_train, y_train)   # Trenowanie modelu
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))

        results.append((name, acc, f1))

    print("\n=== PODSUMOWANIE ===")
    for r in results:
        print(f"{r[0]} | accuracy={r[1]:.3f} | f1={r[2]:.3f}")


if __name__ == "__main__":
    main()
