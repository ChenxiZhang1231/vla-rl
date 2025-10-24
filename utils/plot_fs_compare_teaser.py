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

def _ema(y, alpha=0.2):
    if y is None or len(y) == 0:
        return y
    y = np.array(y, dtype=float)
    m = np.isfinite(y)
    out = y.copy()
    prev = None
    for i, v in enumerate(y):
        if not np.isfinite(v):
            out[i] = prev if prev is not None else np.nan
            continue
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out[i] = prev
    return out

# 一些高质感、色弱友好的 2 色配色（不使用 seaborn）
_PALETTES = {
    # 稳重现代（默认）
    "indigo-teal": ("#3F51B5", "#009688"),
    # 冷静专业
    "slate-emerald": ("#34495E", "#2ECC71"),
    # 学术深色调
    "navy-cerulean": ("#1B3A57", "#2AA1D3"),
    # 暗墨+铜（金属感，打印友好）
    "ink-copper": ("#1F2A37", "#B87333"),
    # 暗夜+薄荷（高对比）
    "charcoal-mint": ("#2F3E46", "#2EC4B6"),
    # Solarized 风
    "blue-cyan": ("#268BD2", "#2AA198"),
}

def plot_method_means(
    res,
    save_path="avg_test.png",
    show=False,
    title="4-task avg (val/test)",
    figsize=6,                     # 正方形：传 6 就是 6x6 inches
    palette="indigo-teal",         # 从 _PALETTES 里选
    linewidth=2.4,                 # 线宽更厚一点
    marker=None,                   # 比如 'o'、'^'；默认不放
    markevery=None,                # 比如 6 或 [10, 30, 60]；默认不放
    markersize=5.5,                # 放 marker 时的大小
    dpi=240,                       # 导出分辨率
):
    base = res.get('baseline', {})
    flow = res.get('flowscale', {})

    s_base = np.asarray(base.get('steps', []), dtype=float)
    y_base = np.asarray(base.get('mean', []), dtype=float)
    s_flow = np.asarray(flow.get('steps', []), dtype=float)
    y_flow = np.asarray(flow.get('mean', []), dtype=float)

    if s_base.size == 0 or s_flow.size == 0:
        raise ValueError("Empty steps in res['baseline'] or res['flowscale'].")

    m_base = np.isfinite(s_base) & np.isfinite(y_base)
    m_flow = np.isfinite(s_flow) & np.isfinite(y_flow)

    # 颜色
    if palette not in _PALETTES:
        raise ValueError(f"Unknown palette '{palette}'. Available: {list(_PALETTES.keys())}")
    c_base, c_flow = _PALETTES[palette]

    # 画布（正方形）
    plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = plt.gca()

    # 线
    ax.plot(
        s_base[m_base], y_base[m_base],
        label="Baseline",
        color=c_base,
        linewidth=linewidth,
        marker=marker, markevery=markevery, markersize=markersize if marker else None,
    )
    ax.plot(
        s_flow[m_flow], y_flow[m_flow],
        label="FlowScale",
        color=c_flow,
        linewidth=linewidth,
        marker=marker, markevery=markevery, markersize=markersize if marker else None,
    )

    # 样式：极简论文风
    ax.set_xlabel("RL training step", fontsize=12)
    ax.set_ylabel("Mean test score (4 tasks)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=15, pad=8)

    # 去掉上/右边框，淡网格
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25)

    # 图例
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"Saved line plot to: {save_path}")
     

def plot_method_means2(
    res,
    save_path="avg_test.png",
    show=False,
    title="4-task avg (val/test)",
    figsize=6,
    palette="indigo-teal",
    linewidth=2.4,
    marker=None,
    markevery=None,
    markersize=5.5,
    dpi=240,
    highlight_max=True,                 # 新增：是否在最大值处打标
    max_marker="o",                     # 新增：最大值标记形状
    max_markersize=70,                  # 新增：scatter 的大小（points^2）
    max_edgecolor="#FFFFFF",            # 新增：白色描边更干净
    max_linewidth=1.6,
    annotate_max=False,                 # 新增：是否标注数值
    annotate_fmt="{y:.3f}",             # 新增：标注格式
):
    import os, numpy as np, matplotlib.pyplot as plt

    _PALETTES = {
        "indigo-teal": ("#3F51B5", "#009688"),
        "slate-emerald": ("#34495E", "#2ECC71"),
        "navy-cerulean": ("#1B3A57", "#2AA1D3"),
        "ink-copper": ("#1F2A37", "#B87333"),
        "charcoal-mint": ("#2F3E46", "#2EC4B6"),
        "blue-cyan": ("#268BD2", "#2AA198"),
    }

    base = res.get('baseline', {})
    flow = res.get('flowscale', {})

    s_base = np.asarray(base.get('steps', []), dtype=float)
    y_base = np.asarray(base.get('mean', []), dtype=float)
    s_flow = np.asarray(flow.get('steps', []), dtype=float)
    y_flow = np.asarray(flow.get('mean', []), dtype=float)

    if s_base.size == 0 or s_flow.size == 0:
        raise ValueError("Empty steps in res['baseline'] or res['flowscale'].")

    m_base = np.isfinite(s_base) & np.isfinite(y_base)
    m_flow = np.isfinite(s_flow) & np.isfinite(y_flow)

    if palette not in _PALETTES:
        raise ValueError(f"Unknown palette '{palette}'. Available: {list(_PALETTES.keys())}")
    c_base, c_flow = _PALETTES[palette]

    plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = plt.gca()

    # 曲线
    ax.plot(s_base[m_base], y_base[m_base], label="Baseline",
            color=c_base, linewidth=linewidth,
            marker=marker, markevery=markevery, markersize=markersize if marker else None)
    ax.plot(s_flow[m_flow], y_flow[m_flow], label="FlowScale",
            color=c_flow, linewidth=linewidth,
            marker=marker, markevery=markevery, markersize=markersize if marker else None)

    # —— 最大值标记 —— #
    if highlight_max:
        # Baseline
        if np.any(m_base):
            yb = y_base[m_base]; sb = s_base[m_base]
            i_max_b = int(np.nanargmax(yb))          # 若有多个最大值，取第一个；要最后一个可改为 np.where(...)[0][-1]
            xb, yb_max = sb[i_max_b], yb[i_max_b]
            ax.scatter([xb], [yb_max], s=max_markersize, marker=max_marker,
                       facecolor=c_base, edgecolor=max_edgecolor, linewidth=max_linewidth, zorder=5)
            if annotate_max:
                ax.annotate(annotate_fmt.format(y=yb_max), (xb, yb_max),
                            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=11)

        # FlowScale
        if np.any(m_flow):
            yf = y_flow[m_flow]; sf = s_flow[m_flow]
            i_max_f = int(np.nanargmax(yf))
            xf, yf_max = sf[i_max_f], yf[i_max_f]
            ax.scatter([xf], [yf_max], s=max_markersize, marker=max_marker,
                       facecolor=c_flow, edgecolor=max_edgecolor, linewidth=max_linewidth, zorder=5)
            if annotate_max:
                ax.annotate(annotate_fmt.format(y=yf_max), (xf, yf_max),
                            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=11)

    # 样式
    ax.set_xlabel("RL training step", fontsize=12)
    ax.set_ylabel("Mean test score (4 tasks)", fontsize=12)
    if title: ax.set_title(title, fontsize=15, pad=8)
    for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved line plot to: {save_path}")
   
def _moving_avg(y, k):
    if k is None or k <= 1:
        return y
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    y_f = y.copy()
    # 仅对有限值做卷积；缺失处保留为原值
    v = np.convolve(y[m], np.ones(k)/k, mode="same")
    y_f[m] = v
    return y_f

def plot_method_means3(
    res,
    save_path="avg_test.png",
    show=False,
    title="4-task avg (val/test)",
    figsize=6,
    palette="indigo-teal",
    linewidth=2.4,
    marker=None,
    markevery=None,
    markersize=5.5,
    dpi=240,
    # 新增：仅用于“线”的可视化平滑；最大值标记仍用原始数据
    smooth_window=1,                 # 1 表示不平滑；常用 5~9
    highlight_max=True,
    max_marker="D",
    max_markersize=70,
    max_edgecolor="#FFFFFF",
    max_linewidth=1.4,
    annotate_max=False,
    annotate_fmt="{y:.3f}",
):
    base = res.get('baseline', {})
    flow = res.get('flowscale', {})

    s_base = np.asarray(base.get('steps', []), dtype=float)
    y_base = np.asarray(base.get('mean', []), dtype=float)
    s_flow = np.asarray(flow.get('steps', []), dtype=float)
    y_flow = np.asarray(flow.get('mean', []), dtype=float)

    if s_base.size == 0 or s_flow.size == 0:
        raise ValueError("Empty steps in res['baseline'] or res['flowscale'].")

    m_base = np.isfinite(s_base) & np.isfinite(y_base)
    m_flow = np.isfinite(s_flow) & np.isfinite(y_flow)

    c_base, c_flow = _PALETTES[palette]

    # 用于“画线”的平滑版
    yb_plot = _moving_avg(y_base[m_base], smooth_window)
    yf_plot = _moving_avg(y_flow[m_flow], smooth_window)

    plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = plt.gca()

    ax.plot(s_base[m_base], yb_plot, label="Baseline", color=c_base,
            linewidth=linewidth, marker=marker, markevery=markevery,
            markersize=markersize if marker else None)
    ax.plot(s_flow[m_flow], yf_plot, label="FlowScale", color=c_flow,
            linewidth=linewidth, marker=marker, markevery=markevery,
            markersize=markersize if marker else None)

    # —— 最大值标记（基于原始数据计算）——
    if highlight_max:
        # baseline
        yb_raw = y_base[m_base]; sb = s_base[m_base]
        if yb_raw.size:
            ib = np.where(yb_raw == np.nanmax(yb_raw))[0][-1]  # 取最后出现的最大值
            xb, yb_max = sb[ib], yb_raw[ib]
            ax.scatter([xb], [yb_max], s=max_markersize, marker=max_marker,
                       facecolor=c_base, edgecolor=max_edgecolor, linewidth=max_linewidth, zorder=5)
            if annotate_max:
                ax.annotate(annotate_fmt.format(y=yb_max), (xb, yb_max),
                            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=11)
        # flowscale
        yf_raw = y_flow[m_flow]; sf = s_flow[m_flow]
        if yf_raw.size:
            iff = np.where(yf_raw == np.nanmax(yf_raw))[0][-1]
            xf, yf_max = sf[iff], yf_raw[iff]
            ax.scatter([xf], [yf_max], s=max_markersize, marker=max_marker,
                       facecolor=c_flow, edgecolor=max_edgecolor, linewidth=max_linewidth, zorder=5)
            if annotate_max:
                ax.annotate(annotate_fmt.format(y=yf_max), (xf, yf_max),
                            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=11)

    # 样式
    ax.set_xlabel("RL training step", fontsize=12)
    ax.set_ylabel("Mean test score (4 tasks)", fontsize=12)
    if title: ax.set_title(title, fontsize=15, pad=8)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
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
    
    # plot_method_means(res, save_path='debug_avg_test.png', show=False, title='4-task avg (val/test)')

    # plot_method_means(res, save_path="teaser_efficiency.png",
    #     title="FlowScale achieves target accuracy with far fewer steps",
    #     palette='indigo-teal')

    # plot_method_means2(res, save_path="avg_max.png",
    #     palette="indigo-teal",
    #     marker=None, highlight_max=True, max_marker="D")  # 菱形
    
    plot_method_means3(res, smooth_window=1, highlight_max=True, annotate_max=True)

if __name__ == "__main__":
    main()
