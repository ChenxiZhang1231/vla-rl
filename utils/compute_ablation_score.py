# We'll compute per-model, per-task mean and variance (sample, ddof=1)
# and also the overall mean/variance across all four tasks for each model.

import numpy as np
import pandas as pd
# from caas_jupyter_tools import display_dataframe_to_user

# Raw data (each cell is 3 runs: seed1/seed2/seed3)
# data = {
#     "1w": {
#         "spatial": [0.8246, 0.8185, 0.8367],
#         "object":  [0.7963, 0.7701, 0.7823],
#         "goal":    [0.7984, 0.8026, 0.8004],
#         "long":    [0.7601, 0.7903, 0.8085],
#     },
#     "flow-grpo-action": {
#         "spatial": [0.9294, 0.9214, 0.9214],
#         "object":  [0.8750, 0.8690, 0.8619],
#         "goal":    [0.8871, 0.8589, 0.8770],
#         "long":    [0.8488, 0.8528, 0.8306],
#     },
#     "fs": {
#         "spatial": [0.9536, 0.9516, 0.9414],
#         "object":  [0.8790, 0.8750, 0.8690],
#         "goal":    [0.9194, 0.9049, 0.9113],
#         "long":    [0.8790, 0.8569, 0.8649],
#     },
# }


data = {
    "1w": {
        "spatial": [0.8246, 0.8185, 0.8367],
        "object":  [0.7963, 0.7701, 0.7823],
        "goal":    [0.7984, 0.8026, 0.8004],
        "long":    [0.7601, 0.7903, 0.8085],
    },
    "flow-grpo-action": {
        "spatial": [0.0, 0.0, 0.0],
        "object":  [0.0, 0.0, 0.0],
        "goal":    [0.0, 0.0, 0.0],
        "long":    [0.8203, 0.0, 0.0],
    },
    "fs": {
        "spatial": [0.8911, 0.879, 0.9011],
        "object":  [0.8286, 0.7944, 0.0],
        "goal":    [0.8488, 0.0, 0.0],
        "long":    [0.8367, 0.0, 0.0],
    },
}

models = list(data.keys())
tasks = ["spatial", "object", "goal", "long"]  # long(10w) labeled "long"

# Compute per-model, per-task stats
rows = []
for model in models:
    for task in tasks:
        vals = np.array(data[model][task], dtype=float)
        mean = vals.mean()
        var = vals.var(ddof=1)  # sample variance
        std = np.sqrt(var)
        rows.append({
            "model": model,
            "task": task,
            "n": len(vals),
            "mean": mean,
            "std": std,
            "var": var,
        })

per_task_df = pd.DataFrame(rows).set_index(["model", "task"]).sort_index()

# Compute per-model overall stats across all four tasks (all 12 points)
overall_rows = []
for model in models:
    all_vals = np.concatenate([np.array(data[model][t], dtype=float) for t in tasks])
    mean = all_vals.mean()
    var = all_vals.var(ddof=1)  # sample variance across all points
    std = np.sqrt(var)
    overall_rows.append({
        "model": model,
        "n": len(all_vals),
        "mean": mean,
        "std": std,
        "var": var,
    })

overall_df = pd.DataFrame(overall_rows).set_index("model").sort_index()

# Round for display
per_task_display = per_task_df.copy()
overall_display = overall_df.copy()
per_task_display[["mean", "std", "var"]] = per_task_display[["mean", "std", "var"]].round(6)
overall_display[["mean", "std", "var"]] = overall_display[["mean", "std", "var"]].round(6)

print(per_task_display)
print()
print(overall_display)
# display_dataframe_to_user("Per-model per-task mean & variance (sample)", per_task_display.reset_index())
# display_dataframe_to_user("Per-model overall (across 4 tasks; 12 points) mean & variance (sample)", overall_display.reset_index())
