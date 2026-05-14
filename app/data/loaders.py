import csv
from pathlib import Path

import pandas as pd


def load_data(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            sample = csv_file.read(4096)
            csv_file.seek(0)

            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ","

        return pd.read_csv(file_path, delimiter=delimiter)

    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(file_path)

    raise ValueError(f"Unsupported file format: {suffix}")
