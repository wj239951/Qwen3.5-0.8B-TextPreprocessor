"""合成「脏文本 → 干净文本」平行语料。

标签由 ``cleaner.normalizer``（规则 oracle）在同一条噪声样本上现场生成，
因此标签与规范严格一致、可复现、零标注成本。

防泄漏：先按**种子**划分 train/valid/test，同一篇种子的所有变体只会落在同一个集合里。

用法：
    python src/data/build_rule_pairs.py --out data/rule --train 12000 --valid 400 --test 400
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cleaner.normalizer import clean  # noqa: E402
from data import noise, seeds  # noqa: E402

LEVELS = ("light", "medium", "heavy")
#: 不同类别组合时的连接符
_JOINERS = ("\n\n", "\n", " ")


def compose(rng: random.Random, pool: list[str]) -> tuple[str, int]:
    """把 1-3 个种子单元拼成一段更长的文本，并返回主种子下标。"""
    k = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    idxs = [rng.randrange(len(pool)) for _ in range(k)]
    if k == 1:
        return pool[idxs[0]], idxs[0]
    parts = [pool[i] for i in idxs]
    joiner = rng.choice(_JOINERS)
    return joiner.join(parts), idxs[0]


def split_seeds(pool: list[str], rng: random.Random) -> dict[str, list[str]]:
    """按种子切分，保证同一篇种子的变体不跨集合。"""
    shuffled = pool[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_valid = max(2, int(n * 0.08))
    n_test = max(2, int(n * 0.08))
    return {
        "test": shuffled[:n_test],
        "valid": shuffled[n_test : n_test + n_valid],
        "train": shuffled[n_test + n_valid :],
    }


def build(split: str, pool: list[str], count: int, rng: random.Random) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    stats: Counter = Counter()
    attempts = 0
    max_attempts = count * 6
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        raw_clean, seed_idx = compose(rng, pool)
        # 两种构造方式：
        #   corrupt  —— 常规多噪声注入（主力）
        #   identity —— 原样保留，教模型"干净文本不要动"（尤其是"刚刚""谢谢"这类叠词）
        #
        # 为什么训练数据里没有乱码：0.8B 模型学不会字节级解码，只会记住"看到乱码就
        # 输出某段中文"，实测会凭空编造整段内容。乱码修复属于确定性代码
        # （`repair_mojibake()`），放在调用链前面做一次即可，不该压给模型。
        mode = rng.choices(("corrupt", "identity"), weights=(0.78, 0.22))[0]
        if mode == "identity":
            noisy, kinds = raw_clean, ["identity"]
            level = "none"
        else:
            level = rng.choices(LEVELS, weights=[0.25, 0.5, 0.25])[0]
            noisy, kinds = noise.corrupt(raw_clean, rng, level)
        label = clean(noisy)
        if not label:
            stats["drop_empty"] += 1
            continue
        if label == noisy and mode != "identity":
            stats["drop_no_change"] += 1
            continue
        row = {
            "raw": noisy,
            "clean": label,
            "split": split,
            "level": level,
            "injectors": kinds,
            "seed_id": f"{split}-{seed_idx}",
            "exact": label == raw_clean,
            "mode": mode,
        }
        rows.append(row)
        stats["kept"] += 1
        stats[f"mode_{mode}"] += 1
        stats[f"level_{level}"] += 1
        stats[f"exact_{row['exact']}"] += 1
        for k in kinds:
            stats[f"inj_{k}"] += 1
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "rule"))
    ap.add_argument("--train", type=int, default=12000)
    ap.add_argument("--valid", type=int, default=400)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261114)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = seeds.flat_seeds()
    parts = split_seeds(pool, rng)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"train": args.train, "valid": args.valid, "test": args.test}
    all_stats: Counter = Counter()
    for split in ("train", "valid", "test"):
        rows, stats = build(split, parts[split], counts[split], rng)
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_stats.update(stats)
        exact = stats.get("exact_True", 0) / max(1, len(rows))
        print(f"[{split}] {len(rows)} 条 -> {path}  完全一致率={exact:.1%}")
        for k in LEVELS:
            print(f"    level={k}: {stats.get(f'level_{k}', 0)}")

    print("\n=== 注入器覆盖 ===")
    for name in noise.INJECTORS:
        print(f"  {name:<16} {all_stats.get(f'inj_{name}', 0)}")
    print(f"\n总保留 {all_stats['kept']} 条，丢弃空标签 {all_stats['drop_empty']}，丢弃无变化 {all_stats['drop_no_change']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
