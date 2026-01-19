# Projekt zaliczeniowy – klasyfikacja pozycji człowieka

## Opis projektu
Celem projektu jest stworzenie programu przetwarzającego dane oraz trenującego modele
uczenia maszynowego do klasyfikacji pozycji człowieka (standing / laying) na podstawie
wektorów cech numerycznych.

## Dane
Dane pochodzą z publicznego zbioru danych dostępnego online i zostały zapisane w formacie CSV.
Zawierają współrzędne punktów charakterystycznych sylwetki oraz etykietę klasy.

## Pipeline
1. Wczytanie danych (DataLoader)
2. Wstępna analiza danych (eda.py)
3. Normalizacja danych (StandardScaler)
4. Podział na zbiór treningowy i testowy
5. Trenowanie modeli klasyfikacyjnych
6. Ewaluacja wyników

## Modele
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

## Wyniki
Modele zostały ocenione za pomocą metryk accuracy oraz F1-score.
Najlepsze wyniki uzyskał model Random Forest.

## Uruchomienie
```bash
pip install -r requirements.txt
python src/main.py
