import torch

path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/work_dirs/merged_ckpts_openvla_fb5_mini128/bridge/carrot/step39.pt"

state = torch.load(path)
print()