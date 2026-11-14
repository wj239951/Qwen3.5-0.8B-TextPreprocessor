"""把大文件作为 GitHub Release 资产流式上传。

为什么不用 curl：`curl --data-binary @file` 会把整个文件读进内存，1.3GB 的权重组
会直接 OOM；`requests` 传文件句柄是流式的，内存占用恒定。

用法：
    $env:GITHUB_TOKEN = "ghp_xxx"
    python tools/upload_release_asset.py --repo owner/name --release-id 123 `
        --file model.zip --name Qwen3.5-0.8B-TextPreprocessor-v1.0.0.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--release-id", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("缺少环境变量 GITHUB_TOKEN", file=sys.stderr)
        return 1

    path = Path(args.file)
    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1
    name = args.name or path.name
    size = path.stat().st_size
    print(f"上传 {name}（{size / 1024 ** 3:.2f} GB）-> {args.repo} release {args.release_id}")

    url = f"https://uploads.github.com/repos/{args.repo}/releases/{args.release_id}/assets"
    started = time.time()
    with path.open("rb") as fh:
        resp = requests.post(
            url,
            params={"name": name},
            data=fh,  # 流式：requests 会按块读取，不会把文件读进内存
            headers={
                "Authorization": f"token {token}",
                "Content-Type": "application/octet-stream",
                "Accept": "application/vnd.github+json",
                "User-Agent": "dsh-agent",
            },
            timeout=(30, 3600),
        )
    elapsed = time.time() - started
    if resp.status_code in (200, 201):
        data = resp.json()
        print(f"成功：{data.get('browser_download_url')}")
        print(f"耗时 {elapsed / 60:.1f} 分钟，平均 {size / 1024 ** 2 / max(elapsed, 1):.2f} MB/s")
        return 0
    print(f"失败 HTTP {resp.status_code}: {resp.text[:400]}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
