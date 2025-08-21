from huggingface_hub import HfApi, CommitOperationAdd
from pathlib import Path

api = HfApi()
repo_id = "jasonzhango/LIBERO-Lerobot"
local_root = Path("/inspire/ssd/project/robotsimulation/public/data/LIBERO-Lerobot/libero_full_lerobot")

# 确保仓库存在
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

ops = []
for subdir in sorted(local_root.iterdir()):
    if subdir.is_dir() and subdir.name.endswith("meta"):
        for f in subdir.rglob("*"):
            if f.is_file():
                print(f"Adding file: {f}")
                path_in_repo = f.relative_to(local_root).as_posix()
                ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(f)))

print(f"Planned upload: {len(ops)} files")
api.create_commit(
    repo_id=repo_id,
    repo_type="dataset",
    operations=ops,
    commit_message="Initial upload of libero* datasets",
    revision="main",
)
print("Upload done.")
