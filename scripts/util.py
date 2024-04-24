import pandas as pd

df = pd.read_json("assets/train_full.jsonl", lines = True)
df = df.sample(frac = 1)
train = df[:35000]
dev = df[35000:]
train.to_json("assets/train.jsonl", orient = "records", lines = True)
dev.to_json("assets/dev.jsonl", orient = "records", lines = True)