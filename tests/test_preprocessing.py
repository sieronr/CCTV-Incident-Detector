import numpy as np
from src.preprocessing import Preprocessor

def test_scaler_shape():
    X = np.array([[1, 2], [3, 4]])
    pre = Preprocessor()
    X_scaled = pre.fit_transform(X)

    assert X.shape == X_scaled.shape
