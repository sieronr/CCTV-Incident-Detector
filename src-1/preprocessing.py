from sklearn.preprocessing import StandardScaler

# Standaryzacja danych wejściowych
class Preprocessor:
    def __init__(self):
        # Inicjalizacja obiektu skalera
        self.scaler = StandardScaler()

# Dopasowanie do danych i skalowanie
    def fit_transform(self, X):
        return self.scaler.fit_transform(X) # Operacja na wektorach (Wektoryzacja danych)

# Skalowanie danych przy użyciu dopasowanego skalera
    def transform(self, X):
        return self.scaler.transform(X)
