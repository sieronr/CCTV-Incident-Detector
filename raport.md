## RAPORT PROJEKTOWY

### 1. Cel projektu
Celem projektu było stworzenie programu przetwarzającego dane oraz budującego model
klasyfikacyjny, który na podstawie cech wejściowych przewiduje klasę decyzyjną.
Projekt realizuje pełny pipeline uczenia maszynowego: od wczytania danych, przez
przetwarzanie, aż do ewaluacji wyników.

---

### 2. Opis danych
Zbiór danych składa się z 121 obserwacji oraz 34 cech numerycznych opisujących współrzędne
punktów charakterystycznych sylwetki człowieka. Zmienna decyzyjna `label` przyjmuje wartości
`standing` lub `laying`. Dane nie zawierają brakujących wartości, a rozkład klas jest niemal
zbalansowany.

---

### 3. Wstępna analiza danych
Wstępna analiza danych obejmowała:
- sprawdzenie struktury danych (`info()`),
- analizę statystyk opisowych (`describe()`),
- analizę rozkładu klas,
- obliczenie macierzy korelacji cech numerycznych.

Analiza wykazała silne zależności pomiędzy współrzędnymi punktów, co uzasadnia zastosowanie
algorytmów klasyfikacyjnych w przestrzeni cech numerycznych.

---

### 4. Przetwarzanie danych
Dane zostały poddane normalizacji z wykorzystaniem algorytmu `StandardScaler`, co zapewnia
porównywalność skali cech oraz poprawia zbieżność algorytmów uczących się.
Normalizacja została wykonana wyłącznie na zbiorze treningowym w celu uniknięcia zjawiska
data leakage.

---

### 5. Budowa i trenowanie modeli
Zastosowano trzy różne algorytmy klasyfikacyjne:
- Logistic Regression jako model bazowy,
- Random Forest jako model zespołowy,
- Support Vector Machine z jądrem RBF.

Dla poszczególnych modeli dobrano hiperparametry w celu poprawy jakości predykcji.

---

### 6. Analiza wyników
Modele oceniono na zbiorze testowym przy użyciu metryk accuracy oraz F1-score.
Ze względu na niemal zbalansowany rozkład klas, metryka F1-score stanowiła wiarygodną miarę
jakości klasyfikacji.

Najlepsze wyniki uzyskał model Random Forest, co sugeruje jego większą odporność na
współzależności pomiędzy cechami.

---

### 7. Testy jednostkowe
Dla kluczowych elementów projektu przygotowano testy jednostkowe, które sprawdzają poprawność
działania modułów wczytywania danych, przetwarzania oraz modeli klasyfikacyjnych.

---

### 8. Wnioski
Projekt spełnia wszystkie założenia zadania i prezentuje kompletny pipeline uczenia
maszynowego. Uzyskane wyniki potwierdzają poprawność stworzonego modelu oraz zasadność
zastosowanych metod przetwarzania danych.