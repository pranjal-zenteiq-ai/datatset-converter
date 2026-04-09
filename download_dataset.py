from datasets import load_dataset
import os
import time

datasets = [
    ("meta-math/MetaMathQA", "metamath"),
    ("ise-uiuc/Magicoder-Evol-Instruct-110K", "magicoder"),
    ("allenai/tulu-v2-sft-mixture", "tulu"),
    ("teknium/OpenHermes-2.5", "openhermes"),
    ("LDJnr/Puffin", "puffin"),
]

os.makedirs("data", exist_ok=True)


def download_with_retry(name, retries=3, delay=5):
    for attempt in range(retries):
        try:
            print(f"Downloading {name} (Attempt {attempt+1})...")
            ds = load_dataset(name, split="train")
            return ds
        except Exception as e:
            print(f"Error: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay}s...\n")
                time.sleep(delay)
            else:
                raise e


for name, short in datasets:
    path = f"data/{short}.parquet"

    if os.path.exists(path):
        print(f"Skipping {name}, already exists.")
        continue

    try:
        ds = download_with_retry(name)
    except:
        ds = load_dataset(name)
        split = list(ds.keys())[0]
        ds = ds[split]

    ds.to_parquet(path)
    print(f"Saved {name} → {path}\n")

print("All datasets downloaded successfully.")