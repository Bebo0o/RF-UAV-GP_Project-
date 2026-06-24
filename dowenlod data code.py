from huggingface_hub import hf_hub_download, login, HfApi
import os

# Login
login(token="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")

repo_id = "BEBO0o/RFUAV"
repo_type = "dataset"

# 👉 CHANGE THIS to D:
BASE_DIR = "D:\gp data\RFUAV_valid"

api = HfApi()
files = api.list_repo_files(repo_id, repo_type=repo_type)

# Filter valid images
valid_files = [f for f in files if f.startswith("ImageSet-AllDrones-MatlabPipeline/valid/")]

print(f"Total files: {len(valid_files)}")

for file in valid_files:
    # create same folder structure inside D:
    relative_path = "/".join(file.split("/")[2:])
    local_path = os.path.join(BASE_DIR, relative_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # ✅ Skip if already exists (IMPORTANT for resume)
    if os.path.exists(local_path):
        print(f"Skipped: {local_path}")
        continue

    hf_hub_download(
        repo_id=repo_id,
        filename=file,
        repo_type=repo_type,
        local_dir=os.path.dirname(local_path),
        local_dir_use_symlinks=False
    )

    print(f"Downloaded: {local_path}")

print("✅ Download completed!")