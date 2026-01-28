from data_loader import DataLoader
from preprocessing import Preprocessor
from models import FallClassifier, RandomForestModel, SVMModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    DATA_PATH = "../data/processed/dataset.csv"


    # Wczytanie danych

    df = DataLoader(DATA_PATH).load()

    X = df.drop(columns=["label"])
    y = df["label"]

    print("\nROZKŁAD KLAS:")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        raise RuntimeError(
            "Dataset ma tylko jedną klasę. Sprawdź build_dataset i folder labels."
        )


    # ROZKŁAD KLAS

    plt.figure(figsize=(6, 4))
    sns.countplot(x=df["label"])
    plt.title("Rozkład klas w zbiorze danych")
    plt.xlabel("Klasa")
    plt.ylabel("Liczba próbek")
    plt.tight_layout()
    plt.show()


    # MACIERZ KORELACJI


    plt.figure(figsize=(12, 8))
    corr = df.drop(columns=["label"]).corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Macierz korelacji cech (keypoints)")
    plt.tight_layout()
    plt.show()


    # Podział na zbiory


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Normalizacja danych


    pre = Preprocessor()
    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)


    # Modele


    models = {
        "Logistic Regression": FallClassifier(),
        "Random Forest": RandomForestModel(),
        "SVM": SVMModel()
    }

    results = []


    # Trenowanie + ewaluacja

    for name, model in models.items():
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"\n=== {name} ===")
        print(classification_report(y_test, y_pred))

        results.append((name, acc, f1))


        # Macierz pomyłek


        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Macierz pomyłek – {name}")
        plt.xlabel("Predykcja")
        plt.ylabel("Prawda")
        plt.tight_layout()
        plt.show()


    # Porównanie pomyłek

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"])

    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=results_df.melt(id_vars="Model"),
        x="Model",
        y="value",
        hue="variable"
    )
    plt.title("Porównanie modeli klasyfikacji")
    plt.ylabel("Wartość metryki")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    print("\n=== PODSUMOWANIE ===")
    for r in results:
        print(f"{r[0]} | accuracy={r[1]:.3f} | f1={r[2]:.3f}")


if __name__ == "__main__":
    main()
