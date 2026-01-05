#!/usr/bin/env python3
"""读取 parquet 文件的 keys 和 shapes"""

import pyarrow.parquet as pq
from pathlib import Path

def read_parquet_info(parquet_path: str):
    pf = pq.ParquetFile(parquet_path)

    print(f"文件: {parquet_path}")
    print(f"行数: {pf.metadata.num_rows}")
    print(f"列数: {pf.metadata.num_columns}")
    print("\nKeys 和 Shapes:")
    print("-" * 50)

    # 读取第一行来获取 shape
    table = pf.read_row_group(0)
    for col_name in table.column_names[:]:  # 空列表表示所有列
        col = table.column(col_name)
        print(f"{col_name:40s} | shape: {col[0].shape if hasattr(col[0], 'shape') else type(col[0]).__name__}")

if __name__ == "__main__":
    parquet_path = "/inspire/ssd/project/robotsimulation/zhangchenxi-253108310322/code/prorl/vla-rl/openpi/bridge_orig/data/chunk-000/episode_000306.parquet"
    read_parquet_info(parquet_path)
