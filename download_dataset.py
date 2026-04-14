from datasets import load_dataset
import os
import time

datasets = [
    ("meta-math/MetaMathQA", "metamathqa"),
    ("ise-uiuc/Magicoder-Evol-Instruct-110K", "magicoder_evol_instruct"),
    ("allenai/tulu-v2-sft-mixture", "tulu_v2_sft_mixture"),
    ("teknium/OpenHermes-2.5", "openhermes_2_5"),
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