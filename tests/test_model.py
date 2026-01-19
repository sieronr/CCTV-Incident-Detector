import numpy as np
from src.models import FallClassifier


def test_model_can_train_and_predict():
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = ["laying", "standing", "laying"]

    model = FallClassifier()
    model.train(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
