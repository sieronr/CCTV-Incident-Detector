from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Trenowanie modelu

# Odzielanie danych

class FallClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)  # Fine tuning

    def train(self, X, y):
        self.model.fit(X, y)    # Trenowanie

    def predict(self, X):
        return self.model.predict(X)

# Model zespołowy

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

    def train(self, X, y):
        self.model.fit(X, y)    # Trenowanie

    def predict(self, X):
        return self.model.predict(X)

# Maszyna wektorów nośnych

class SVMModel:
    def __init__(self):
        self.model = SVC(kernel="rbf")

    def train(self, X, y):
        self.model.fit(X, y)    # Trenowanie

    def predict(self, X):
        return self.model.predict(X)
