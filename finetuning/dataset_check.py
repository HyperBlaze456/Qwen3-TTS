import pandas as pd
from datasets import load_dataset

gi_ds = load_dataset("simon3000/genshin-voice", split="train", streaming=True,)
sr_ds = load_dataset("simon3000/starrail-voice", split="train", streaming=True,)

gi_test = gi_ds.take(3)
sr_test = sr_ds.take(3)

for i, example in enumerate(gi_test):
    print(f"Genshin, Row {i}: {example}")

for i, example in enumerate(sr_test):
    print(f"Starrail, Row {i}: {example}")
