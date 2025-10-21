#!/usr/bin/env bash
# bash shell/merge_to_target_cp.sh /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_total --dry-run
# bash shell/merge_to_target_cp.sh /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ /inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/wm_data_process/WM-data-processed/LIBERO-WM-WXYZ-ADDI-VLA-Adapter/libero_total --dry-run

set -euo pipefail

usage() {
  echo "Usage: $0 SRC_ROOT DEST_DIR [--dry-run]"
  echo "Example:"
  echo "  $0 /path/to/LIBERO-WM-WXYZ-ADDI-VLA-Adapter /path/to/MERGED_OUT"
  exit 1
}

[[ $# -ge 2 ]] || usage
SRC_ROOT="$(realpath "$1")"
DEST_DIR="$(realpath -m "$2")"
[[ -d "$SRC_ROOT" ]] || { echo "Not a directory: $SRC_ROOT"; exit 1; }

DRY=0
if [[ "${3:-}" == "--dry-run" ]]; then DRY=1; fi

# 需要合并的标准目录名（若不存在则跳过）
MERGE_DIRS=(blacks clips metadata overlays resampled temp_videos)

do_cp() {
  # $1: src  $2: dst
  local src="$1" dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ $DRY -eq 1 ]]; then
    echo "[DRY] cp -p \"$src\" \"$dst\""
  else
    cp -p "$src" "$dst"
  fi
}

copy_tree_with_conflict_prefix() {
  # $1: src_dir  $2: dst_dir  $3: prefix
  local src_dir="$1" dst_dir="$2" prefix="$3"
  # 逐文件复制；保持目录结构；冲突则在文件名加前缀
  find "$src_dir" -type f -print0 | while IFS= read -r -d '' f; do
    local rel="${f#"$src_dir/"}"
    local dst="$dst_dir/$rel"
    mkdir -p "$(dirname "$dst")"
    if [[ -e "$dst" ]]; then
      dst="$(dirname "$dst")/${prefix}__$(basename "$dst")"
    fi
    do_cp "$f" "$dst"
  done
}

echo "SRC_ROOT: $SRC_ROOT"
echo "DEST_DIR: $DEST_DIR"
mkdir -p "$DEST_DIR"

# 目标里的统一去向（标准目录、散落文件、其余目录）
for d in "${MERGE_DIRS[@]}"; do mkdir -p "$DEST_DIR/$d"; done
mkdir -p "$DEST_DIR/_merged_files"

shopt -s nullglob
for child in "$SRC_ROOT"/*/; do
  child_base="$(basename "$child")"

  # 跳过目标目录自身（若恰好放在SRC_ROOT里）
  [[ "$DEST_DIR" == "$child"* ]] && continue

  echo "==> From child: $child_base"

  # 1) 合并标准目录到目标目录对应的同名目录
  for d in "${MERGE_DIRS[@]}"; do
    if [[ -d "$child/$d" ]]; then
      echo "  - copying $child_base/$d -> $DEST_DIR/$d"
      copy_tree_with_conflict_prefix "$child/$d" "$DEST_DIR/$d" "$child_base"
    fi
  done

  # 2) 子目录根部散落文件 -> 目标/_merged_files（加前缀）
  find "$child" -maxdepth 1 -type f -print0 | while IFS= read -r -d '' f; do
    fname="$(basename "$f")"
    dst="$DEST_DIR/_merged_files/${child_base}__${fname}"
    do_cp "$f" "$dst"
  done

  # 3) 其余非标准目录 -> 目标的同名目录（若冲突逐文件加前缀）
  find "$child" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' sub; do
    sub_base="$(basename "$sub")"
    # 已处理过的标准目录跳过
    for d in "${MERGE_DIRS[@]}"; do
      [[ "$sub_base" == "$d" ]] && continue 2
    done
    # 复制到目标的同名目录
    echo "  - copying extra dir: $child_base/$sub_base -> $DEST_DIR/$sub_base"
    mkdir -p "$DEST_DIR/$sub_base"
    copy_tree_with_conflict_prefix "$sub" "$DEST_DIR/$sub_base" "$child_base"
  done
done

echo "Done."
