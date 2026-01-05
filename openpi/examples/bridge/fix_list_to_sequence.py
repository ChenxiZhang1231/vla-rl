#!/usr/bin/env python3
"""
将 LeRobot 数据集 parquet 文件中的 List 类型改为 Sequence 类型
使其兼容 datasets 3.x
"""

import shutil
from pathlib import Path
import pyarrow.parquet as pq


def fix_dataset(source_dir: Path, output_dir: Path):
    """复制数据集并修改 parquet metadata 中的 List 为 Sequence"""

    print(f"源目录: {source_dir}")
    print(f"输出目录: {output_dir}")

    # 清理输出目录
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # 复制 meta 和 images 目录
    if (source_dir / "meta").exists():
        shutil.copytree(source_dir / "meta", output_dir / "meta")
    if (source_dir / "images").exists():
        print("  复制 images 目录...")
        shutil.copytree(source_dir / "images", output_dir / "images")

    # 创建 data 目录
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    # 处理 parquet 文件
    data_dir = source_dir / "data"
    output_data_dir = output_dir / "data"

    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        print(f"  处理: {chunk_dir.name}")
        (output_data_dir / chunk_dir.name).mkdir(exist_ok=True)

        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            print(f"    {parquet_file.name}")

            # 读取 parquet 文件
            pf = pq.ParquetFile(parquet_file)
            table = pf.read()

            # 获取原始 metadata
            metadata = table.schema.metadata
            if b"huggingface" in metadata:
                hf_meta = metadata[b"huggingface"].decode()
                # 把 List 改成 Sequence
                old = hf_meta
                hf_meta_fixed = hf_meta.replace('"_type": "List"', '"_type": "Sequence"')
                metadata[b"huggingface"] = hf_meta_fixed.encode()

                if old != hf_meta_fixed:
                    print(f"      已修改: List -> Sequence")
                else:
                    print(f"      无需修改")

                # 创建新的 schema 并重新写入
                new_schema = table.schema.with_metadata(metadata)
                new_table = table.cast(new_schema)

                # 写入新文件
                output_file = output_data_dir / chunk_dir.name / parquet_file.name
                pq.write_table(new_table, output_file, compression="snappy")
            else:
                # 没有 huggingface metadata，直接复制
                output_file = output_data_dir / chunk_dir.name / parquet_file.name
                pq.write_table(table, output_file, compression="snappy")

    print("\n完成!")


if __name__ == "__main__":
    source = Path("/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/code/prorl/vla-rl/openpi1")
    output = Path("/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/code/prorl/vla-rl/openpi1_fixed")

    fix_dataset(source, output)
