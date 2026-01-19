import pandas as pd

# Wczytywanie danych z pliku CSV
class DataLoader:
    def __init__(self, path):
        # Przechowanie ścieżki do pliku z danymi
        self.path = path

    # Wczytywanie danych z pliku CSV i zwrócenie ich jako DataFrame
    def load(self):
        return pd.read_csv(self.path)
