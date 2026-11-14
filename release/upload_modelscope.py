"""把模型目录推送到 ModelScope（魔搭）。

准备：
    pip install modelscope
    $env:MODELSCOPE_API_TOKEN = "你的访问令牌"   # https://www.modelscope.cn/my/myaccesstoken

用法：
    python release/upload_modelscope.py --dir Qwen3.5-0.8B-TextPreprocessor \
        --repo Qwen3.5-0.8B-TextPreprocessor

踩过的坑：
* visibility 的取值是 **private=1 / internal=3 / public=5**，传 1 会得到一个私有仓库；
  本脚本默认建公开（5），若仓库已存在会顺手把可见性纠正过来。
* 上传 1.6GB 单文件耗时约 12 分钟，中途断线重跑会做增量。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PUBLIC = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="要上传的模型目录")
    ap.add_argument("--repo", required=True, help="仓库名，如 Qwen3.5-0.8B-TextPreprocessor")
    ap.add_argument("--namespace", default="", help="留空则用当前账号")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--message", default="上传蒸馏模型")
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

    visibility = 1 if args.private else PUBLIC
    api = HubApi()
    api.login(token)

    # 命名空间：SDK 没有直接取用户名的接口，从环境变量或参数给
    namespace = args.namespace or os.environ.get("MODELSCOPE_NAMESPACE", "")
    if not namespace:
        print("请用 --namespace 指定账号名（或设 MODELSCOPE_NAMESPACE）", file=sys.stderr)
        return 1
    repo_id = f"{namespace}/{args.repo}"

    try:
        url = api.create_model(repo_id, visibility=visibility, license="Apache-2.0")
        print(f"已创建: {url}")
    except Exception as exc:  # 已存在
        print(f"仓库已存在或创建失败（继续尝试上传）: {type(exc).__name__}: {exc}")

    # 兜底：确保可见性正确（create_model 的 visibility 在部分版本里不生效）
    try:
        api.update_model(repo_id, visibility=visibility)
        print(f"可见性已设为 {'私有' if args.private else '公开'}")
    except Exception as exc:  # noqa: BLE001
        print(f"可见性设置失败（可在网页上手动改）: {type(exc).__name__}: {exc}")

    files = [p for p in local.iterdir() if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    print(f"上传 {len(files)} 个文件，共 {size / 1024 ** 3:.2f} GB")
    api.upload_folder(repo_id=repo_id, folder_path=str(local), commit_message=args.message)
    print(f"完成：https://www.modelscope.cn/models/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
