import pandas as pd

files = [
    "data/metamath.parquet",
    "data/magicoder.parquet",
    "data/tulu.parquet",
    "data/openhermes.parquet",
]

for f in files:
    print(f"\n===== {f} =====")
    df = pd.read_parquet(f)
    print(df.columns)
    print(df.head(2))