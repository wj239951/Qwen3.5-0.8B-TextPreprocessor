"""通过 GitHub API 推送本地仓库（绕过被阻断的 git 端点）。

为什么需要它：在部分网络环境下 `github.com:443` 的 git 端点被阻断（push 报
`Connection was reset`），而且 `/git/blobs`、`/git/trees` 也会被中间设备挡成
409/422。实测 **Contents API（PUT /contents）和 /git/commits 可用**，因此本脚本：

1. 逐个文件用 Contents API 写入（每个文件会生成一个临时提交）；
2. 读取最终 tree，用 `/git/commits` 创建一个 `parents=[]` 的提交；
3. 把分支强制指向它 —— 于是远端只剩**一个**干净提交。

用法：
    $env:GITHUB_TOKEN = "ghp_xxx"
    python tools/push_via_api.py --repo owner/name --branch main

注意：
* 只推送 **已 git 跟踪** 的文件（按 .gitignore 过滤），权重不会被推上去；
* 单文件超过 100MB 会失败，权重请走 Release / ModelScope。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"


def api(method: str, path: str, token: str, payload: dict | None = None, ok_body: bool = True) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "dsh-agent",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if (ok_body and body.strip()) else {}


def try_api(method: str, path: str, token: str, payload: dict | None = None) -> dict | None:
    try:
        return api(method, path, token, payload)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="")
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-squash", action="store_true", help="不压缩历史（保留逐文件提交）")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("缺少环境变量 GITHUB_TOKEN", file=sys.stderr)
        return 1

    root = Path(args.root).resolve()
    files = tracked_files(root)
    if not files:
        print("没有已跟踪文件", file=sys.stderr)
        return 1

    message = args.message or subprocess.run(
        ["git", "log", "-1", "--pretty=%B"], cwd=root, capture_output=True
    ).stdout.decode("utf-8").strip()

    print(f"推送 {len(files)} 个文件到 {args.repo}@{args.branch}")
    for idx, rel in enumerate(files, 1):
        abs_path = root / rel
        if not abs_path.is_file():
            continue
        content = abs_path.read_bytes()
        if len(content) > 100 * 1024 * 1024:
            print(f"  拒绝：{rel} 超过 100MB", file=sys.stderr)
            return 1
        payload = {
            "message": f"add {rel}",
            "content": base64.b64encode(content).decode("ascii"),
            "branch": args.branch,
        }
        existing = try_api("GET", f"/repos/{args.repo}/contents/{urllib.parse.quote(rel)}?ref={args.branch}", token)
        if existing and isinstance(existing, dict) and existing.get("sha"):
            payload["sha"] = existing["sha"]
        api("PUT", f"/repos/{args.repo}/contents/{urllib.parse.quote(rel)}", token, payload)
        if idx % 10 == 0 or idx == len(files):
            print(f"  已写入 {idx}/{len(files)}")

    if args.no_squash:
        print(f"完成：https://github.com/{args.repo}/tree/{args.branch}")
        return 0

    ref = api("GET", f"/repos/{args.repo}/git/ref/heads/{args.branch}", token)
    head = ref["object"]["sha"]
    commit = api("GET", f"/repos/{args.repo}/git/commits/{head}", token)
    tree_sha = commit["tree"]["sha"]
    squashed = api("POST", f"/repos/{args.repo}/git/commits", token, {
        "message": message,
        "tree": tree_sha,
        "parents": [],
    })
    api("PATCH", f"/repos/{args.repo}/git/refs/heads/{args.branch}", token, {
        "sha": squashed["sha"],
        "force": True,
    })
    print(f"已把 {args.branch} 指向单个提交 {squashed['sha'][:12]}")
    print(f"完成：https://github.com/{args.repo}/tree/{args.branch}")
    return 0


if __name__ == "__main__":
    import urllib.parse  # noqa: E402  （放在末尾避免顶部导入顺序干扰）

    raise SystemExit(main())
