import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(repo_id="hiyouga/geometry3k",
                  repo_type="dataset",
                  local_dir="/data/datasets/geometry3k",
                  local_dir_use_symlinks=False,
                  resume_download=True)