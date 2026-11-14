"""从逐条预测文件里挑代表性样例，生成 markdown 对照表。

给模型卡和 README 用：既要展示"清洗得对"，也要展示"错在哪"——只贴成功案例的
模型卡不值得信任。

用法：
    python tools/make_examples.py --preds eval/preds_test.jsonl --out examples.md --n 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def esc(text: str, limit: int = 120) -> str:
    """压成单行、转义竖线，方便塞进表格。

    脏文本里可能带控制字符（\x00-\x1f），必须剔除，否则生成的文件会被当成二进制。
    """
    text = "".join(ch for ch in text if ch >= " " or ch == "\n")
    text = text.replace("\r", "").replace("\n", "\\n")
    text = text.replace("|", "\\|")
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", default=str(ROOT / "examples.md"))
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.preds).open(encoding="utf-8") if line.strip()]
    hits = [r for r in rows if r["pred"].strip() == r["ref"].strip()]
    misses = [r for r in rows if r["pred"].strip() != r["ref"].strip()]

    # 挑"噪声种类多"的成功案例，和"差异最大"的失败案例
    hits.sort(key=lambda r: -len(r["raw"]))
    misses.sort(key=lambda r: -abs(len(r["pred"]) - len(r["ref"])) - (r["pred"] != r["ref"]))

    lines = ["# 清洗效果示例", "", f"共 {len(rows)} 条测试样本，完全一致 {len(hits)} 条 "
             f"({len(hits) / max(1, len(rows)):.1%})。", "", "## 成功案例", "",
             "| 原始输入 | 模型输出 |", "|---|---|"]
    for r in hits[: args.n]:
        lines.append(f"| `{esc(r['raw'])}` | `{esc(r['pred'])}` |")

    lines += ["", "## 失败案例（差异最大）", "", "| 原始输入 | 标准答案 | 模型输出 |", "|---|---|---|"]
    for r in misses[: args.n]:
        lines.append(f"| `{esc(r['raw'])}` | `{esc(r['ref'])}` | `{esc(r['pred'])}` |")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"写入 {args.out}（成功 {min(len(hits), args.n)} 条，失败 {min(len(misses), args.n)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
