"""把 release 目录推送到 ModelScope（魔搭）。

准备：
    pip install modelscope
    $env:MODELSCOPE_API_TOKEN = "你的访问令牌"   # https://www.modelscope.cn/my/myaccesstoken

用法：
    python release/upload_modelscope.py --dir release/Qwen3.5-0.8B-TextPreprocessor \
        --repo Qwen3.5-0.8B-TextPreprocessor
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="要上传的模型目录")
    ap.add_argument("--repo", required=True, help="仓库名，如 Qwen3.5-0.8B-TextPreprocessor")
    ap.add_argument("--namespace", default="", help="留空则用当前账号")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--message", default="upload distilled text preprocessor")
    args = ap.parse_args()

    token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if not token:
        print("缺少环境变量 MODELSCOPE_API_TOKEN", file=sys.stderr)
        return 1
    local = Path(args.dir)
    if not local.is_dir():
        print(f"目录不存在: {local}", file=sys.stderr)
        return 1

    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        print("请先 pip install modelscope", file=sys.stderr)
        return 1

    api = HubApi()
    api.login(token)
    namespace = args.namespace or api.get_user_name()
    repo_id = f"{namespace}/{args.repo}"
    print(f"创建/确认仓库 {repo_id}")
    api.create_model(repo_id, visibility=0 if args.private else 1)
    print(f"上传 {local} -> {repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(local),
        commit_message=args.message,
    )
    print(f"完成：https://www.modelscope.cn/models/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
