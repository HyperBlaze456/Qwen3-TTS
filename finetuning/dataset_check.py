import sys
from pathlib import Path

import pandas as pd


def main():
    data_dir = Path("./dataset")

    if not data_dir.exists():
        print(f"Error: {data_dir} directory not found")
        sys.exit(1)

    parquet_files = list(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        sys.exit(1)

    for pf in parquet_files:
        print(f"\n{'='*60}")
        print(f"File: {pf}")
        print(f"{'='*60}")

        df = pd.read_parquet(pf)
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nColumn dtypes:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")

        print(f"\nFirst row sample:")
        for col in df.columns:
            val = df[col].iloc[0]
            val_str = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
            print(f"  {col}: {val_str}")


if __name__ == "__main__":
    main()
