from datasets import load_dataset
import os
import time

datasets = [
    ("totally-not-an-llm/EverythingLM-data-V3", "everythinglm"),
    ("microsoft/orca-math-word-problems-200k", "orca"),
    ("qintongli/GSM-Plus-v0", "gsm"),
    ("Vezora/Tested-143k-Python-Alpaca", "python_alpaca"),
    ("LDJnr/Puffin", "puffin"),
    ("roneneldan/TinyStories", "tinystories"),
    ("nvidia/HelpSteer2", "nemotron"),
    ("allenai/tulu-3-sft-mixture", "tulu3"),
    ("infly/OpenCoder-Instruction", "opencoder"),
]

os.makedirs("data", exist_ok=True)


def download_with_retry(name, retries=3, delay=5):
    for attempt in range(retries):
        try:
            print(f"Downloading {name} (Attempt {attempt+1})...")
            # For some datasets we might need a specific split or subset
            if name == "nvidia/HelpSteer2":
               # HelpSteer2 usually requires handling
               ds = load_dataset(name, split="train")
            else:
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
    except Exception as e:
        print(f"Standard download failed for {name}: {e}. Trying full load...")
        try:
            ds = load_dataset(name)
            split = list(ds.keys())[0]
            ds = ds[split]
        except Exception as e2:
            print(f"Failed to load {name} even with full load: {e2}. Skipping.")
            continue

    ds.to_parquet(path)
    print(f"Saved {name} → {path}\n")

print("All datasets downloaded successfully.")