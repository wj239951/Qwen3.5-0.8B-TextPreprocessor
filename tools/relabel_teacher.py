"""离线重标注老师数据。

规则引擎改动后，已采到的老师样本需要用新 oracle 重新打标，避免"数据与规范脱节"。
本脚本只读本地 JSONL，不再调用老师模型。

用法：
    python tools/relabel_teacher.py --in data/teacher/gen.jsonl --out data/teacher/gen_v2.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cleaner.normalizer import clean  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "data" / "teacher" / "gen.jsonl"))
    ap.add_argument("--out", dest="out", default=str(ROOT / "data" / "teacher" / "gen_v2.jsonl"))
    ap.add_argument("--min-similarity", type=float, default=0.90)
    args = ap.parse_args()

    src = Path(args.inp)
    dst = Path(args.out)
    kept = dropped = 0
    reasons: Counter = Counter()
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = obj["raw"]
            teacher_clean = (obj.get("teacher_clean") or obj["clean"]).strip()
            oracle = clean(raw)
            if not oracle:
                reasons["oracle 为空"] += 1
                dropped += 1
                continue
            if oracle == raw:
                reasons["raw 已无需清洗"] += 1
                dropped += 1
                continue
            sim = difflib.SequenceMatcher(None, oracle, teacher_clean).ratio()
            if sim < args.min_similarity:
                reasons["与 oracle 分歧过大"] += 1
                dropped += 1
                continue
            obj["clean"] = oracle
            obj["similarity"] = round(sim, 4)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"保留 {kept} 条 -> {dst}")
    print(f"丢弃 {dropped} 条")
    for reason, cnt in reasons.most_common():
        print(f"  {cnt:>4}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
