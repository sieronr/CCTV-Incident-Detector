import os
import pandas as pd
import random


class DatasetBuilder:
    def __init__(self, labels_dir: str, output_csv: str):
        self.labels_dir = labels_dir
        self.output_csv = output_csv

    def build(self):
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"Folder nie istnieje: {self.labels_dir}")

        rows = []

        for file in os.listdir(self.labels_dir):
            if not file.endswith(".txt"):
                continue

            path = os.path.join(self.labels_dir, file)

            with open(path, "r") as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) < 8:
                        continue

                    keypoints = values[5:]

                    features = []
                    for i in range(0, len(keypoints) - 2, 3):
                        x = keypoints[i]
                        y = keypoints[i + 1]
                        features.extend([x, y])

                    rows.append(features)

        if not rows:
            raise RuntimeError("Brak danych w labels!")

        #  DODANIE SZTUCZNEJ KLASY
        random.shuffle(rows)
        half = len(rows) // 2

        labeled_rows = []
        for i, feat in enumerate(rows):
            label = "laying" if i < half else "standing"
            labeled_rows.append(feat + [label])

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        num_kp = len(labeled_rows[0]) - 1
        columns = []
        for i in range(1, num_kp // 2 + 1):
            columns += [f"kp{i}_x", f"kp{i}_y"]
        columns.append("label")

        df = pd.DataFrame(labeled_rows, columns=columns)
        df.to_csv(self.output_csv, index=False)

        print("Dataset utworzony (SZTUCZNE KLASY)")
        print(df["label"].value_counts())


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    builder = DatasetBuilder(
        labels_dir=os.path.join(BASE_DIR, "data", "raw", "labels"),
        output_csv=os.path.join(BASE_DIR, "data", "processed", "dataset.csv"),
    )

    builder.build()
