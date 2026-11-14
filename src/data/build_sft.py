"""把规则合成数据与老师数据合并成 SFT 训练集。

输出 JSONL，每行 ``{"messages": [system, user, assistant], "src": "...", "id": "..."}``。
同一 ``id`` 只保留一条；老师样本按 ``--teacher-repeat`` 加权，避免被规则数据淹没。

用法：
    python src/data/build_sft.py --rule data/rule/train.jsonl --teacher data/teacher/gen.jsonl \
        --out data/sft/train.jsonl --teacher-repeat 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.format import SYSTEM_PROMPT  # noqa: E402
from cleaner.normalizer import repair_mojibake  # noqa: E402

MAX_CHARS = 4000  # 超长样本留到评测里单独看，训练里截断过狠会影响质量


def has_mojibake(text: str) -> bool:
    """判断样本里是否含可修复的乱码。

    含乱码的样本一律不进训练集：小模型学不会字节级还原，只会记住"看到乱码就输出
    某段中文"，实测会凭空编造内容。乱码由 repair_mojibake() 在调用链里处理。
    """
    return repair_mojibake(text)[0] != text


def load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        print(f"  [skip] {path} 不存在")
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "raw" in obj and "clean" in obj:
                rows.append(obj)
    return rows


def to_record(pair: dict, src: str) -> dict:
    raw, clean = pair["raw"], pair["clean"]
    return {
        "id": hashlib.sha1((raw + "\x00" + clean).encode("utf-8")).hexdigest()[:16],
        "src": src,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw},
            {"role": "assistant", "content": clean},
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", nargs="*", default=[str(ROOT / "data" / "rule" / "train.jsonl")])
    ap.add_argument("--teacher", nargs="*", default=[str(ROOT / "data" / "teacher" / "gen.jsonl")])
    ap.add_argument("--teacher-repeat", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "data" / "sft" / "train.jsonl"))
    ap.add_argument("--valid-out", default=str(ROOT / "data" / "sft" / "valid.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    records: list[dict] = []
    seen: set[str] = set()
    stats: Counter = Counter()

    for path in args.rule:
        pairs = load_pairs(Path(path))
        kept = 0
        for pair in pairs:
            if len(pair["raw"]) > args.max_chars:
                stats["too_long"] += 1
                continue
            if has_mojibake(pair["raw"]):
                stats["mojibake_drop"] += 1
                continue
            rec = to_record(pair, "rule")
            if rec["id"] in seen:
                stats["dup"] += 1
                continue
            seen.add(rec["id"])
            records.append(rec)
            kept += 1
        stats["rule"] += kept
        print(f"  rule  {path}: {kept}")

    for path in args.teacher:
        pairs = load_pairs(Path(path))
        kept = 0
        for pair in pairs:
            if len(pair["raw"]) > args.max_chars:
                stats["too_long"] += 1
                continue
            if has_mojibake(pair["raw"]):
                stats["mojibake_drop"] += 1
                continue
            rec = to_record(pair, "teacher")
            if rec["id"] in seen:
                stats["dup"] += 1
                continue
            seen.add(rec["id"])
            for _ in range(max(1, args.teacher_repeat)):
                records.append(rec)
                kept += 1
        stats["teacher"] += kept
        print(f"teacher {path}: {kept}（repeat={args.teacher_repeat}）")

    if not records:
        print("没有可用样本", file=sys.stderr)
        return 1

    rng.shuffle(records)
    # 8% 留作验证集，避免只用规则验证集评估老师数据
    n_valid = max(64, int(len(records) * 0.08))
    valid, train = records[:n_valid], records[n_valid:]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in train:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    vout = Path(args.valid_out)
    vout.parent.mkdir(parents=True, exist_ok=True)
    with vout.open("w", encoding="utf-8") as fh:
        for rec in valid:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n训练集 {len(train)} 条 -> {out}")
    print(f"验证集 {len(valid)} 条 -> {vout}")
    print(f"来源分布: rule={stats['rule']}, teacher={stats['teacher']}, 去重={stats['dup']}, "
          f"超长丢弃={stats['too_long']}, 乱码丢弃={stats['mojibake_drop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
