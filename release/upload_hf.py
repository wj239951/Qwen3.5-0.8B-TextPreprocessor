"""把 release 目录推送到 Hugging Face Hub。

准备：
    pip install huggingface_hub
    $env:HF_TOKEN = "hf_xxx"

用法：
    python release/upload_hf.py --dir release/Qwen3.5-0.8B-TextPreprocessor \
        --repo your-name/Qwen3.5-0.8B-TextPreprocessor
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repo", required=True, help="形如 user/model-name")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--message", default="upload distilled text preprocessor")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("缺少环境变量 HF_TOKEN", file=sys.stderr)
        return 1
    local = Path(args.dir)
    if not local.is_dir():
        print(f"目录不存在: {local}", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("请先 pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    print(f"创建/确认仓库 {args.repo}")
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print(f"上传 {local} -> {args.repo}")
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=str(local),
        commit_message=args.message,
    )
    print(f"完成：https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
