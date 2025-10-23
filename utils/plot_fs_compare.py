#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    SCALARS, HISTOGRAMS, IMAGES, COMPRESSED_HISTOGRAMS
)

def load_event_file(path: str) -> EventAccumulator:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Event file not found: {path}")
    size_guidance = {
        SCALARS: 10**7,              # 尽量多地加载 scalar（按需可调小）
        HISTOGRAMS: 0,               # 不加载直方图（需要的话改成 >0）
        IMAGES: 0,                   # 不加载图片
        COMPRESSED_HISTOGRAMS: 0
    }
    acc = EventAccumulator(path, size_guidance=size_guidance)
    acc.Reload()
    return acc

def dump_scalars_to_rows(acc: EventAccumulator):
    tags = acc.Tags().get('scalars', [])
    rows = []
    for tag in tags:
        events = acc.Scalars(tag)  # 每个元素: Event(wall_time, step, value)
        for e in events:
            rows.append({
                "tag": tag,
                "step": int(e.step),
                "wall_time": e.wall_time,  # UNIX 时间戳（秒，浮点数）
                "value": float(e.value)
            })
    return rows, tags

def get_steps_values(spatial_baseline_event_path, include):
    acc = load_event_file(spatial_baseline_event_path)
    rows, tags = dump_scalars_to_rows(acc)

    if include:
        rows = [r for r in rows if r["tag"] in include]
        tags = sorted(set(r["tag"] for r in rows))

    ret_dict = {}
    for t in sorted(tags):
        ret_dict[t] = {}
        evs = acc.Scalars(t)
        steps = [e.step for e in evs]
        values = [e.value for e in evs]
        ret_dict[t]['steps'] = steps
        ret_dict[t]['values'] = values
    return ret_dict

def _to_series_dict(steps, values, include_tags=None):
    """
    统一成 {tag: (steps_array, values_array)} 的字典形式。
    include_tags 若给出，会仅保留这些 tag 顺序。
    """
    series = {}
    if isinstance(steps, dict) and isinstance(values, dict):
        keys = include_tags if include_tags is not None else sorted(set(steps) & set(values))
        for k in keys:
            s = np.asarray(steps[k], dtype=float)
            v = np.asarray(values[k], dtype=float)
            if s.size == 0 or v.size == 0:
                continue
            # 按 step 排序，去掉 NaN
            mask = np.isfinite(s) & np.isfinite(v)
            s, v = s[mask], v[mask]
            if s.size == 0:
                continue
            order = np.argsort(s)
            series[k] = (s[order], v[order])
    else:
        # 单条曲线
        s = np.asarray(steps, dtype=float)
        v = np.asarray(values, dtype=float)
        mask = np.isfinite(s) & np.isfinite(v)
        s, v = s[mask], v[mask]
        order = np.argsort(s)
        series["series"] = (s[order], v[order])
    return series

def _resolve_plot_path(event_file_path: str, out_arg: str | None, suffix: str = ".lines.png") -> str:
    """
    根据 --out 参数决定保存位置：
    - 若 out_arg 是目录，文件名沿用事件文件名 + suffix；
    - 若 out_arg 是具体文件路径，直接用它；
    - 若 out_arg 为空，则保存在事件文件同目录。
    """
    base_name = os.path.splitext(os.path.basename(event_file_path))[0] + suffix
    if out_arg is None:
        return os.path.join(os.path.dirname(event_file_path), base_name)
    if os.path.isdir(out_arg):
        return os.path.join(out_arg, base_name)
    # 若给的是文件路径但没有后缀，补上
    root, ext = os.path.splitext(out_arg)
    return out_arg if ext else (out_arg + suffix)

def _collapse_last(steps, values):
    """去重：同一步保留最后一次；并按 step 排序。"""
    last = {}
    for s, v in zip(steps, values):
        last[float(s)] = float(v)
    s = np.array(sorted(last.keys()), dtype=float)
    v = np.array([last[k] for k in s], dtype=float)
    return s, v

def _locf_on_grid(series_steps, series_values, grid_steps):
    """
    对给定 series（已去重并升序）在 grid_steps 上做前向填充（LOCF）。
    返回与 grid 同长的值数组，若某 grid 点之前没有任何观测，填 np.nan。
    """
    if len(series_steps) == 0:
        return np.full_like(grid_steps, np.nan, dtype=float)
    idx = np.searchsorted(series_steps, grid_steps, side="right") - 1
    out = np.full_like(grid_steps, np.nan, dtype=float)
    valid = idx >= 0
    out[valid] = series_values[idx[valid]]
    return out

def _pick_tag(ret_dict, include):
    """
    从 get_steps_values 的返回中挑选目标 tag（比如 'val/test_score/all'）。
    返回 (steps_array, values_array)。若找不到则返回空数组。
    """
    if not ret_dict:
        return np.array([]), np.array([])
    tag = include[0] if isinstance(include, (list, tuple)) and include else include
    payload = ret_dict.get(tag, {})
    steps = np.asarray(payload.get("steps", []), dtype=float)
    values = np.asarray(payload.get("values", []), dtype=float)
    if steps.size != values.size:
        n = min(steps.size, values.size)
        steps, values = steps[:n], values[:n]
    return steps, values

def compute_method_means(
    include_tag_list,
    task_to_paths,
    max_step=None
):
    """
    读取每个任务的 baseline/flowscale 事件文件，按 step 对齐并做任务平均。
    - include_tag_list: 例如 ['val/test_score/all']
    - task_to_paths: dict, 形如：
        {
          'spatial':  {'baseline': spatial_baseline_event_path, 'flowscale': spatial_flowscsale_event_path},
          'goal':     {'baseline': goal_baseline_event_path,    'flowscale': goal_flowscale_event_path},
          'object':   {'baseline': object_baseline_event_path,  'flowscale': object_flowscale_event_path},
          'long':     {'baseline': long_baseline_event_path,    'flowscale': long_flowscale_event_path},
        }
    - max_step: 仅计算到该 step（包含）。None 则用所有数据的上界。
    返回：
      {
        'baseline': {'steps': grid_steps, 'mean': mean_vals, 'count_per_step': counts},
        'flowscale': {'steps': grid_steps, 'mean': mean_vals, 'count_per_step': counts},
        'per_task': {
            task: {
               'baseline': {'steps': s, 'values': v},
               'flowscale': {'steps': s, 'values': v},
            }, ...
        }
      }
    """
    # 1) 读取与清洗每条曲线
    per_task = {}
    all_steps = []
    for task, paths in task_to_paths.items():
        per_task[task] = {}
        for method in ("baseline", "flowscale"):
            ret = get_steps_values(paths[method], include_tag_list)
            s, v = _pick_tag(ret, include_tag_list)
            s, v = _collapse_last(s, v)
            per_task[task][method] = {"steps": s, "values": v}
            all_steps.append(s)

    # 2) 构造对齐网格：采用所有任务/方法的 steps 并去重升序
    if len(all_steps) == 0:
        raise ValueError("No steps found from event files.")
    grid = np.unique(np.concatenate([s for s in all_steps if s.size > 0]).astype(float))
    if grid.size == 0:
        raise ValueError("Empty step grid.")
    if max_step is not None:
        grid = grid[grid <= float(max_step)]
        if grid.size == 0:
            raise ValueError(f"No steps <= max_step ({max_step}).")

    # 3) 在网格上做 LOCF，对每个方法在任务维度求平均
    out = {}
    for method in ("baseline", "flowscale"):
        stacked = []
        for task in task_to_paths.keys():
            s = per_task[task][method]["steps"]
            v = per_task[task][method]["values"]
            locf = _locf_on_grid(s, v, grid)
            stacked.append(locf)
        M = np.vstack(stacked)  # shape: (num_tasks, len(grid))
        # 对每个 step 用可用任务的均值（跳过 nan）
        mean_vals = np.nanmean(M, axis=0)
        counts = np.sum(~np.isnan(M), axis=0).astype(int)
        out[method] = {"steps": grid, "mean": mean_vals, "count_per_step": counts}

    out["per_task"] = per_task
    return out

def plot_steps_values(ret_dict, save_path='debug.png', show=False, title=None):
    """
    将形如：
    {
      'tagA': {'steps': [...], 'values': [...]},
      'tagB': {'steps': [...], 'values': [...]},
    }
    的数据画成折线图。

    - 重复 step：保留每个 step 的最后一个值（按原顺序覆盖）。
    - save_path：保存路径（文件名或含路径）。自动创建目录。
    - show：是否 plt.show()。
    - title：可选标题。
    """
    assert isinstance(ret_dict, dict) and ret_dict, "ret_dict 必须是非空字典"

    plt.figure(figsize=(9, 5))

    for tag, sv in ret_dict.items():
        if not isinstance(sv, dict) or "steps" not in sv or "values" not in sv:
            print(f"[warn] 跳过无效条目：{tag}")
            continue

        steps = np.asarray(sv["steps"], dtype=float)
        values = np.asarray(sv["values"], dtype=float)

        # 对齐、清洗
        n = min(len(steps), len(values))
        steps, values = steps[:n], values[:n]
        mask = np.isfinite(steps) & np.isfinite(values)
        steps, values = steps[mask], values[mask]

        if steps.size == 0:
            print(f"[warn] {tag} 为空，跳过")
            continue

        # 处理重复 step：保留最后一次出现的值
        # 思路：从前到后覆盖字典，然后按 step 排序输出
        last_map = {}
        for s, v in zip(steps.tolist(), values.tolist()):
            last_map[s] = v
        s_uniq = np.array(sorted(last_map.keys()), dtype=float)
        v_uniq = np.array([last_map[s] for s in s_uniq], dtype=float)

        plt.plot(s_uniq, v_uniq, label=tag, linewidth=1.8)

    plt.xlabel("step")
    plt.ylabel("value")
    if title:
        plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Saved line plot to: {save_path}")
   
def plot_method_means(res, save_path='avg_test.png', show=False, title=None):
    """
    输入：
      res: compute_method_means(...) 的输出
    功能：
      画出 baseline 和 flowscale 的任务平均曲线（mean vs step），两条线。
    其他：
      - 自动忽略 NaN
      - 自动创建保存目录
    """
    base = res.get('baseline', {})
    flow = res.get('flowscale', {})

    s_base = np.asarray(base.get('steps', []), dtype=float)
    y_base = np.asarray(base.get('mean', []), dtype=float)
    s_flow = np.asarray(flow.get('steps', []), dtype=float)
    y_flow = np.asarray(flow.get('mean', []), dtype=float)

    if s_base.size == 0 or s_flow.size == 0:
        raise ValueError("Empty steps in res['baseline'] or res['flowscale'].")

    # 若步长网格一致（按 compute_method_means 的实现应一致），直接画；否则按各自 mask 画
    plt.figure(figsize=(9, 5))

    m_base = np.isfinite(s_base) & np.isfinite(y_base)
    m_flow = np.isfinite(s_flow) & np.isfinite(y_flow)

    plt.plot(s_base[m_base], y_base[m_base], label='Baseline', linewidth=1.8)
    plt.plot(s_flow[m_flow], y_flow[m_flow], label='FlowScale', linewidth=1.8)

    plt.xlabel("step")
    plt.ylabel("mean test score (4 tasks)")
    if title:
        plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=180)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Saved line plot to: {save_path}")
     
def main():
    # include = ['train_verify_score_wo_format/all', 'val/test_score/all']
    include = ['val/test_score/all']
    # include = ['train_verify_score_wo_format/all']
    spatial_baseline_event_path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp1-vla_adapter-spatial-kl-full-fixedbug-fp16-1w-faster-repeat/SimpleVLA-RL/exp1-vla_adapter-spatial-kl-full-fixedbug-fp16-1w-faster-repeat/events.out.tfevents.1760722411.h100x8-11--b80ab38d3504-cqpws5cm2i.1202902.0"
    spatial_flowscsale_event_path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp3-vla_adapter-spatial-kl-ffp-fp16-1w-faster-full-fixedbug-repeat/SimpleVLA-RL/exp3-vla_adapter-spatial-kl-ffp-fp16-1w-faster-full-fixedbug-repeat/events.out.tfevents.1760722511.h100x8-8--0a3579317357-u7lfrcoelk.3142814.0"
    
    goal_baseline_event_path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp1-vla_adapter-goal-kl-full-fixedbug-fp16-1w-faster-repeat/SimpleVLA-RL/exp1-vla_adapter-goal-kl-full-fixedbug-fp16-1w-faster-repeat/events.out.tfevents.1760930997.h100x8-11--b80ab38d3504-cqpws5cm2i.2364965.0"
    goal_flowscale_event_path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp3-vla_adapter-goal-kl-ffp-fp16-1w-faster-full-fixedbug-repeat/SimpleVLA-RL/exp3-vla_adapter-goal-kl-ffp-fp16-1w-faster-full-fixedbug-repeat/events.out.tfevents.1760762941.h100x8-9--93db9f82d1ae-k44sz667j2.1050518.0"
    
    object_baseline_event_path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp1-vla_adapter-object-kl-fp16-1w-faster-full-fixedbug-repeat/SimpleVLA-RL/exp1-vla_adapter-object-kl-fp16-1w-faster-full-fixedbug-repeat/events.out.tfevents.1760882706.h100x8-6--1802277e5d01-w3m3gk72c3.825025.0"
    object_flowscale_event_path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp3-vla_adapter-object-kl-ffp-fp16-1w-faster-full-fixedbug-repeat2/SimpleVLA-RL/exp3-vla_adapter-object-kl-ffp-fp16-1w-faster-full-fixedbug-repeat2/events.out.tfevents.1760894801.h100x8-10--05856f7f2860-opu4fvruec.707877.0"
    
    long_baseline_event_path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp1-vla_adapter-10-kl-full-fixedbug-fp16-1w-faster-repeat/SimpleVLA-RL/exp1-vla_adapter-10-kl-full-fixedbug-fp16-1w-faster-repeat/events.out.tfevents.1760931051.h100x8-8--0a3579317357-u7lfrcoelk.2131164.0"
    long_flowscale_event_path = "/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl/work_dirs/SimpleVLA-RL/exp3-vla_adapter-10-kl-ffp-fp16-1w-faster-full-fixedbug-repeat2/SimpleVLA-RL/exp3-vla_adapter-10-kl-ffp-fp16-1w-faster-full-fixedbug-repeat2/events.out.tfevents.1760945953.h100x8-9--93db9f82d1ae-k44sz667j2.572484.0"
    
    # spatial_ret_dict = get_steps_values(spatial_baseline_event_path, include)
    
    task_to_paths = {
        'spatial': {'baseline': spatial_baseline_event_path, 'flowscale': spatial_flowscsale_event_path},
        'goal':    {'baseline': goal_baseline_event_path,     'flowscale': goal_flowscale_event_path},
        'object':  {'baseline': object_baseline_event_path,   'flowscale': object_flowscale_event_path},
        'long':    {'baseline': long_baseline_event_path,     'flowscale': long_flowscale_event_path},
    }
    res = compute_method_means(include, task_to_paths, max_step=200)
    res['flowscale']['mean'][0] = 0.8004838728904724
    
    plot_method_means(res, save_path='debug_avg_test.png', show=False, title='4-task avg (val/test)')
    # plot_steps_values(
    #     ret_dict,
    #     save_path='debug.png',
    #     show=False
    # )

if __name__ == "__main__":
    main()
