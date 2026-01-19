import pandas as pd

# Wstępna analiza danych

DATA_PATH = "../data/processed/dataset.csv"

def main():
    df = pd.read_csv(DATA_PATH) # Wczytanie danych

    print("=== PODSTAWOWE INFORMACJE O DANYCH ===")
    df.info()   # Podstawowe informacje o danych

    print("\n=== PIERWSZE 5 WIERSZY ===")
    print(df.head())    # Podgląd danych

    print("\n=== STATYSTYKI OPISOWE (CECHY NUMERYCZNE) ===")
    print(df.describe())    # Statystyki opisowe cech numerycznych

    print("\n=== ROZKŁAD KLASY DECYZYJNEJ ===")
    print(df["label"].value_counts())   # Rozkład klasy decyzyjnej

    print("\n=== MACIERZ KORELACJI (CECHY NUMERYCZNE) ===")
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()    # Macierz korelacji cech numerycznych

    if "label" in corr.columns:
        print(corr["label"].sort_values(ascending=False))   # Korelacja cech z klasą decyzyjną
    else:
        print(corr)

if __name__ == "__main__":
    main()
