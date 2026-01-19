from src.data_loader import DataLoader


def test_data_loader_loads_data():
    loader = DataLoader("data/processed/dataset.csv")
    df = loader.load()

    assert df is not None
    assert not df.empty
    assert "label" in df.columns
