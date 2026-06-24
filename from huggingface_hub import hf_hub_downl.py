from huggingface_hub import hf_hub_download, login, HfApi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🔥 IMPORTANT (prevent C: from filling)
os.environ["HF_HOME"] = "D:/hf_cache"

# Login
login(token="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")

repo_id = "BEBO0o/RFUAV"
repo_type = "dataset"

# Save to D:
BASE_DIR = r"D:\gp data\RFUAV_valid"

api = HfApi()
files = api.list_repo_files(repo_id, repo_type=repo_type)

# Filter valid images
valid_files = [f for f in files if f.startswith("ImageSet-AllDrones-MatlabPipeline/valid/")]

print(f"Total files: {len(valid_files)}")

# ✅ Download function
def download_file(file):
    try:
        relative_path = "/".join(file.split("/")[2:])
        local_path = os.path.join(BASE_DIR, relative_path)

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Skip if already exists
        if os.path.exists(local_path):
            return f"Skipped: {relative_path}"

        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            repo_type=repo_type,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False
        )

        return f"Downloaded: {relative_path}"

    except Exception as e:
        return f"Error: {file} → {e}"

# ⚡ Parallel execution
MAX_WORKERS = 12  # try 8 first

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_file, f) for f in valid_files]

    for future in as_completed(futures):
        print(future.result())

print("✅ Download completed!")