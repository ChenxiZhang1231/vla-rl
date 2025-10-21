#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
libero_captions_index.py

用法：
# 1) 生成索引 JSON（默认扫描 benchmark.get_benchmark_dict() 里所有可用的 suite）
python libero_captions_index.py build --out libero_captions_index.json

# 仅生成指定 suite（逗号分隔）
python libero_captions_index.py build --suites LIBERO_10,LIBERO_SPATIAL --out libero_captions_index.json

# 2) 读取 JSON，按 caption 反查 suite（先精确后模糊）
python libero_captions_index.py resolve --index libero_captions_index.json --caption "put the red mug on the rack"
"""

import argparse
import json
import sys
import datetime
import difflib

# 可选：无 tqdm 时自动降级
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---- 兼容导入 LIBERO benchmark ----
try:
    from libero.libero import benchmark  # 常见路径
except Exception:
    try:
        import benchmark  # 或者直接是 benchmark
    except Exception as e:
        print("无法导入 LIBERO benchmark 模块。请确认环境中已安装/可导入 'libero.benchmark' 或 'benchmark'。")
        raise e


def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def normalize_caption(text: str) -> str:
    """简单归一化：小写 + 压缩空白。"""
    return " ".join(text.lower().strip().split())


def try_extract_caption(task) -> str:
    """
    从 task 对象上尽最大可能提取 caption。
    兼容常见属性/方法名；若都失败返回空串。
    """
    # 先试属性
    attr_candidates = [
        "caption", "task_caption", "description", "task_description",
        "goal_description", "prompt", "name", "title"
    ]
    for k in attr_candidates:
        if hasattr(task, k):
            v = getattr(task, k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    # 再试可调用方法
    method_candidates = [
        "get_caption", "get_task_caption", "get_description",
        "get_task_description", "get_goal_description", "caption", "name"
    ]
    for k in method_candidates:
        fn = getattr(task, k, None)
        if callable(fn):
            try:
                v = fn()
                if isinstance(v, str) and v.strip():
                    return v.strip()
            except Exception:
                pass

    # 最后试 task.info / task.meta 中的典型键
    for container_key in ["info", "meta", "config", "attrs"]:
        if hasattr(task, container_key):
            d = getattr(task, container_key)
            if isinstance(d, dict):
                for key in ["caption", "task_caption", "description", "goal", "goal_description"]:
                    v = d.get(key, "")
                    if isinstance(v, str) and v.strip():
                        return v.strip()

    return ""  # 实在拿不到就返回空


def build_index_json(out_path: str, suites_csv: str | None):
    bench_dict = benchmark.get_benchmark_dict()

    # 选择要扫描的 suites
    if suites_csv:
        # 用户手动指定（兼容大小写）
        selected = [s.strip() for s in suites_csv.split(",") if s.strip()]
        # 校验键是否存在（用原始键名匹配，不强行转换）
        suite_keys = []
        for name in selected:
            if name in bench_dict:
                suite_keys.append(name)
            else:
                # 尝试大小写宽松匹配
                matches = [k for k in bench_dict.keys() if k.lower() == name.lower()]
                if matches:
                    suite_keys.extend(matches)
                else:
                    print(f"[警告] 未找到 suite: {name}（跳过）")
        if not suite_keys:
            print("[错误] 指定的 suites 在 benchmark 字典中都不存在。")
            sys.exit(1)
    else:
        # 全部
        suite_keys = list(bench_dict.keys())

    by_suite = {}
    caption_to_suite = {}

    for suite_name in suite_keys:
        try:
            task_suite = bench_dict[suite_name]()
        except Exception as e:
            print(f"[跳过] 无法实例化 suite={suite_name}: {e}")
            continue

        try:
            n_tasks = int(task_suite.n_tasks)
        except Exception:
            # 兜底：尝试 len(task_suite)
            try:
                n_tasks = len(task_suite)
            except Exception:
                print(f"[跳过] suite={suite_name} 无法获取任务数量。")
                continue

        suite_items = []
        for task_id in tqdm(range(n_tasks), desc=f"[{suite_name}] tasks"):
            try:
                task = task_suite.get_task(task_id)
            except Exception as e:
                print(f"  [跳过] suite={suite_name} task_id={task_id} 无法 get_task: {e}")
                continue

            # cap = try_extract_caption(task)
            cap = task[1]
            if not cap:
                cap = f""  # 留空也纳入，以便可后续补全

            suite_items.append({"task_id": task_id, "caption": cap})

            # 建立反向索引（只收录非空 caption）
            if cap:
                norm = normalize_caption(cap)
                # 若同一规范化 caption 在不同 suite 重复，记录为列表以避免冲突
                if norm in caption_to_suite:
                    prev = caption_to_suite[norm]
                    if isinstance(prev, list):
                        if suite_name not in prev:
                            prev.append(suite_name)
                    else:
                        if prev != suite_name:
                            caption_to_suite[norm] = [prev, suite_name]
                else:
                    caption_to_suite[norm] = suite_name

        by_suite[suite_name] = suite_items

    payload = {
        "generated_at": now_iso(),
        "libero_version": getattr(benchmark, "__version__", None),
        "by_suite": by_suite,
        "caption_to_suite": caption_to_suite
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[完成] 已写入：{out_path}")


def resolve_suite(index_path: str, caption: str, fuzzy: bool = True, cutoff: float = 0.84):
    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    c2s = idx.get("caption_to_suite", {})
    norm = normalize_caption(caption)

    # 1) 精确匹配（规范化后）
    if norm in c2s:
        print(c2s[norm])
        return

    # 2) 模糊匹配（可返回一个或多个 suite）
    if fuzzy and c2s:
        keys = list(c2s.keys())
        # 使用 difflib 做相似度匹配
        # 获取最相近的若干候选
        candidates = difflib.get_close_matches(norm, keys, n=5, cutoff=cutoff)
        if candidates:
            # 展示候选及其对应 suite
            out = []
            for k in candidates:
                out.append({"matched_caption": k, "suite": c2s[k]})
            print(json.dumps({"match_type": "fuzzy", "candidates": out}, ensure_ascii=False, indent=2))
            return

    # 3) 未命中
    print(json.dumps({"match_type": "none", "message": "No suite match found for the given caption."}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Build/resolve LIBERO caption→suite 索引 JSON")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="生成索引 JSON")
    p_build.add_argument("--out", type=str, required=True, help="输出 JSON 路径")
    p_build.add_argument("--suites", type=str, default=None,
                         help="只处理这些 suite（逗号分隔，名称需与 benchmark.get_benchmark_dict() 的键一致，大小写不敏感）")

    p_resolve = sub.add_parser("resolve", help="读取索引 JSON 并按 caption 反查 suite")
    p_resolve.add_argument("--index", type=str, required=True, help="索引 JSON 路径")
    p_resolve.add_argument("--caption", type=str, required=True, help="要查询的 caption")
    p_resolve.add_argument("--no-fuzzy", action="store_true", help="禁用模糊匹配")
    p_resolve.add_argument("--cutoff", type=float, default=0.84, help="模糊匹配相似度阈值（0~1）")

    args = parser.parse_args()

    if args.cmd == "build":
        build_index_json(args.out, args.suites)
    elif args.cmd == "resolve":
        resolve_suite(args.index, args.caption, fuzzy=(not args.no_fuzzy), cutoff=args.cutoff)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
