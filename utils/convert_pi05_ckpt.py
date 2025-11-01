#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from collections import OrderedDict, defaultdict

import torch

try:
    from safetensors.torch import load_file as st_load, save_file as st_save
    HAS_ST = True
except Exception:
    HAS_ST = False


def load_state(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        if not HAS_ST:
            raise RuntimeError("请先安装 safetensors：pip install safetensors")
        sd = st_load(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        # torch.load 可能返回包含 state_dict 的包装
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
    # 统一成普通 dict[str, Tensor]
    return OrderedDict(sd)


def save_state_safetensors(sd: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not out_path.endswith(".safetensors"):
        out_path = out_path + ".safetensors"
    if not HAS_ST:
        raise RuntimeError("请先安装 safetensors：pip install safetensors")
    st_save(sd, out_path)
    return out_path


def remap_key(k: str) -> str:
    """
    依据目前日志推断的规则进行重命名：
      - 加前缀 'model.'（若缺失）
      - 将 '.dense.weight/.dense.bias' 折叠为 '.weight/.bias'
      - 去掉 'paligemma_with_expert.paligemma.model.' 中多余的 '.model.'
    你可以按需在此补充更多规则。
    """
    newk = k

    # 规则3：去掉 paligemma 路径里多余的 ".model."
    newk = newk.replace(
        "paligemma_with_expert.paligemma.model.",
        "paligemma_with_expert.paligemma."
    )
    newk = newk.replace(
        "paligemma_with_expert.gemma_expert.model.layers",
        "paligemma_with_expert.gemma_expert.model.layers"
    )  # 占位：目前 gemma_expert 这段保持不变，仅清理 dense 后缀

    # 规则2：把 *.dense.{weight,bias} 改成 *.{weight,bias}
    newk = newk.replace(".dense.weight", ".weight")
    newk = newk.replace(".dense.bias", ".bias")

    # 规则1：补上顶层 'model.' 前缀
    if not newk.startswith("model."):
        newk = "model." + newk

    return newk


def main():
    ap = argparse.ArgumentParser(description="Remap PI05 state_dict keys.")
    ap.add_argument("--in", dest="inp", default="/inspire/hdd/project/robotsimulation/public/models/openpi/pi05_bridge/pi05_bridge_1028_01/100000/model.safetensors", help="输入权重路径（.safetensors / .pt / .pth）")
    ap.add_argument("--out", dest="outp", default="/inspire/hdd/project/robotsimulation/public/models/openpi/pi05_bridge/pi05_bridge_1028_01/100000_convert/model.safetensors", help="输出 .safetensors 路径")
    ap.add_argument("--dry", action="store_true", help="仅打印映射效果，不保存")
    # 可选：用于后验校验
    ap.add_argument("--lerobot-src", default="/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/lerobot_pi/src", help="lerobot_pi/src 的路径（可选）")
    ap.add_argument("--saved-model-path", default="/inspire/hdd/project/robotsimulation/public/models/openpi/pi05_bridge/pi05_bridge_1028_01/100000_convert", help="PI05Policy 的目录，用于校验（可选）")
    args = ap.parse_args()

    sd = load_state(args.inp)

    mapped = OrderedDict()
    collisions = defaultdict(list)
    changed = []
    unchanged = []

    for k, v in sd.items():
        nk = remap_key(k)
        if nk in mapped:
            # 冲突：记录并以最后一次为准（或改为 assert/报错）
            collisions[nk].append(k)
        mapped[nk] = v
        if nk != k:
            changed.append((k, nk))
        else:
            unchanged.append(k)

    print(f"[Summary] 原始参数: {len(sd)} 个")
    print(f"[Summary] 映射后参数: {len(mapped)} 个")
    print(f"[Summary] 被改名的参数: {len(changed)} 个；未改名: {len(unchanged)} 个")
    if collisions:
        print(f"[Warning] 发生 {len(collisions)} 处 key 冲突（相同新 key 多次映射），以下仅展示前 5 处：")
        for i, (nk, olds) in enumerate(collisions.items()):
            if i >= 5:
                print("  ...")
                break
            print(f"  新key: {nk}")
            print(f"    来自旧key: {olds}")

    print("\n[Examples] 映射示例（前 12 条）：")
    for i, (ok, nk) in enumerate(changed[:12]):
        print(f"  {i:02d}: {ok}  ->  {nk}")

    if not args.dry:
        out_path = save_state_safetensors(mapped, args.outp)
        print(f"\n[OK] 已保存为: {out_path}")

    # ========= 可选：加载校验 =========
    if args.lerobot_src and args.saved_model_path:
        try:
            sys.path.insert(0, args.lerobot_src)
            from lerobot.policies.pi05 import PI05Policy  # type: ignore

            # 严格校验：先构建，再尝试 load_state_dict
            # 这里不直接 from_pretrained 以避免又去加载原始权重
            # 若你的 PI05Policy 只能 from_pretrained，则可先实例化再覆盖权重：
            policy = PI05Policy.from_pretrained(args.saved_model_path, strict=False)
            missing, unexpected = policy.load_state_dict(mapped, strict=False)
            # 兼容 torch 两种返回形式
            if isinstance(missing, (list, tuple)) and isinstance(unexpected, (list, tuple)):
                missing_keys, unexpected_keys = list(missing), list(unexpected)
            else:
                r = policy.load_state_dict(mapped, strict=False)
                missing_keys, unexpected_keys = r.missing_keys, r.unexpected_keys

            print(f"\n[Check] missing keys: {len(missing_keys)}; unexpected keys: {len(unexpected_keys)}")
            if missing_keys:
                for s in missing_keys[:20]:
                    print("  MISSING:", s)
                if len(missing_keys) > 20:
                    print("  ...")
            if unexpected_keys:
                for s in unexpected_keys[:20]:
                    print("  UNEXPECTED:", s)
                if len(unexpected_keys) > 20:
                    print("  ...")

        except Exception as e:
            print(f"\n[Check] 校验阶段未成功（可忽略或按需调整）：{e}")


if __name__ == "__main__":
    main()
