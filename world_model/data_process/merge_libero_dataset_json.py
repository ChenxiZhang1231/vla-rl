#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
仅在根目录的下一级子目录中查找 dataset.json（或 data.json），并合并。
默认 depth=1；如需更深可传 --depth 2（最多做浅层 BFS，不做全量递归）。

用法：
python merge_one_level.py \
  --roots /path/to/LIBERO-WM-WXYZ /path/to/LIBERO-WM-WXYZ-ADDI \
  --out merged_dataset.json \
  [--also-match data.json] [--add-source] [--depth 1]
"""

import argparse, json, sys
from pathlib import Path
from collections import deque
from typing import List, Union, Dict, Any, Tuple

Json = Union[dict, list]

def bfs_children_dirs(root: Path, max_depth: int) -> List[Path]:
    """返回从 root 开始、深度<=max_depth 的目录（不含 root 自身）。"""
    found = []
    q = deque([(root, 0)])
    while q:
        cur, d = q.popleft()
        if d == max_depth:
            continue
        for p in sorted(cur.iterdir()):
            if p.is_dir():
                found.append(p)
                q.append((p, d + 1))
    return found

def find_jsons_one_level(roots: List[str], patterns: List[str], depth: int) -> List[Path]:
    files = []
    for r in roots:
        root = Path(r).resolve()
        if not root.exists():
            print(f"[警告] 根目录不存在：{root}", file=sys.stderr)
            continue
        # 仅在 depth 层目录中查找
        for subdir in bfs_children_dirs(root, max_depth=depth):
            for pat in patterns:
                fp = subdir / pat
                if fp.exists():
                    files.append(fp.resolve())
    # 去重并排序
    return sorted(set(files))

def annotate_source(obj: Json, source_dir: str) -> Json:
    if isinstance(obj, list):
        out = []
        for it in obj:
            if isinstance(it, dict):
                x = dict(it)
                x.setdefault("_source_dir", source_dir)
                out.append(x)
            else:
                out.append(it)
        return out
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, list):
                out[k] = annotate_source(v, source_dir)
            else:
                out[k] = v
        return out
    return obj

def merge_pair(a: Json, b: Json, hint_a: str, hint_b: str) -> Json:
    # list + list
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    # dict + dict（仅拼接顶层 list 字段，其他冲突保留 a）
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, vb in b.items():
            if k not in out:
                out[k] = vb
                continue
            va = out[k]
            if isinstance(va, list) and isinstance(vb, list):
                out[k] = va + vb
            else:
                # 其他情况保留 a
                out[k] = va
        return out
    # 结构不同：保留 a
    return a

def merge_all(files: List[Path], add_source: bool) -> Tuple[Json, List[str]]:
    merged: Json = []
    stats = []
    for i, fp in enumerate(files, 1):
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[跳过] 读取失败：{fp}  原因：{e}", file=sys.stderr)
            continue
        if add_source:
            data = annotate_source(data, str(fp.parent))

        if i == 1:
            merged = data
        else:
            merged = merge_pair(merged, data, "merged", str(fp))

        # 统计
        if isinstance(data, list):
            stats.append(f"{fp}: list({len(data)})")
        elif isinstance(data, dict):
            list_keys = {k: len(v) for k, v in data.items() if isinstance(v, list)}
            stats.append(f"{fp}: dict(list_keys={list_keys})")
        else:
            stats.append(f"{fp}: {type(data).__name__}")
    return merged, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="根目录（如 LIBERO-WM-WXYZ 和 LIBERO-WM-WXYZ-ADDI）")
    ap.add_argument("--out", required=True, help="输出 JSON")
    ap.add_argument("--also-match", nargs="*", default=[], help="额外文件名（默认只找 dataset.json），如 data.json")
    ap.add_argument("--add-source", action="store_true", help="为样本添加 _source_dir 字段")
    ap.add_argument("--depth", type=int, default=1, help="向下查找层数（默认 1 层）")
    args = ap.parse_args()

    patterns = ["dataset.json"] + [p for p in args.also_match if p != "dataset.json"]

    files = find_jsons_one_level(args.roots, patterns, depth=args.depth)
    if not files:
        print("[错误] 没有找到任何 JSON 文件。请检查根目录与层数。", file=sys.stderr)
        sys.exit(1)

    print(f"[信息] 将合并 {len(files)} 个文件：")
    for fp in files:
        print("  -", fp)

    merged, stats = merge_all(files, add_source=args.add_source)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] 已写入：{out_path}")
    print("[统计]")
    for s in stats:
        print("  *", s)

if __name__ == "__main__":
    main()

