from huggingface_hub import HfApi, CommitOperationAdd
from pathlib import Path

# --- 1. 配置您的信息 ---
# 初始化 HfApi
api = HfApi()

# 设置您的Hugging Face模型仓库ID
# 格式应为 "your-username/your-model-name" 或 "your-org/your-model-name"
repo_id = "jasonzhango/finetune-smolvla-libero-bug" 

# 设置您本地模型文件所在的根目录
local_model_path = Path("/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/jasonzhang/vla-rl/internvl_chat/work_dirs/smolvla-0.5b-ft_expert-bf16-20ep-libero_full/checkpoint-53216")

# --- 2. 创建仓库并准备上传文件 ---
# 确保Hugging Face Hub上的仓库存在，如果不存在则创建
# repo_type="model" 指定这是一个模型仓库
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# 创建一个列表，用于存放所有待上传文件的操作
operations = []

print(f"开始扫描目录 {local_model_path} 下的文件...")

# 使用 rglob("*") 递归遍历本地模型目录下的所有文件和文件夹
for f in local_model_path.rglob("*"):
    # 确保路径是文件，并且文件名不是以 .pt 结尾
    if f.is_file() and not f.name.endswith(".pt"):
        # 打印将要添加的文件路径
        print(f"准备添加文件: {f}")
        
        # 计算文件在仓库中的相对路径
        path_in_repo = f.relative_to(local_model_path).as_posix()
        
        # 创建一个添加文件的操作，并加入到操作列表中
        operations.append(
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(f))
        )

# --- 3. 执行上传 ---
# 检查是否有文件需要上传
if not operations:
    print("没有找到需要上传的文件（已忽略 .pt 文件）。")
else:
    print(f"\n计划上传 {len(operations)} 个文件。")
    
    # 使用 create_commit 一次性将所有文件操作提交到Hugging Face Hub
    api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Upload model files (excluding .pt)", # 您可以自定义提交信息
        revision="main", # 指定要推送到的分支
    )
    
    print("\n上传完成！")