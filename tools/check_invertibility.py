"""可逆性自检：逐个注入器验证 clean(corrupt(x)) == clean(x)。

这是数据质量的守门员测试。某个注入器的不一致率偏高时，说明噪声超出了规范
能逆转的范围，标签会带上系统性偏差，必须修注入器或修规范。

用法：
    python tools/check_invertibility.py --samples 40 --repeats 20
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cleaner.normalizer import clean  # noqa: E402
from data import noise, seeds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show", type=int, default=3, help="每个注入器展示几个反例")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = seeds.flat_seeds()
    picks = rng.sample(pool, min(args.samples, len(pool)))

    print(f"{'injector':<16}{'ok':>6}{'bad':>6}{'保真率':>10}")
    print("-" * 40)
    worst: list[tuple[str, float]] = []
    for name, fn in noise.INJECTORS.items():
        ok = bad = 0
        examples: list[tuple[str, str, str]] = []
        for text in picks:
            base = clean(text)
            for _ in range(args.repeats):
                try:
                    noisy = fn(text, rng)
                except Exception as exc:  # noqa: BLE001
                    bad += 1
                    continue
                got = clean(noisy)
                if got == base:
                    ok += 1
                else:
                    bad += 1
                    if len(examples) < args.show:
                        examples.append((text, got, base))
        rate = ok / max(1, ok + bad)
        print(f"{name:<16}{ok:>6}{bad:>6}{rate:>9.1%}")
        worst.append((name, rate))
        for src, got, exp in examples:
            print(f"    - 源: {src[:40]!r}")
            print(f"      得: {got[:60]!r}")
            print(f"      期: {exp[:60]!r}")

    print("\n最差三个：", ", ".join(f"{n}={r:.0%}" for n, r in sorted(worst, key=lambda x: x[1])[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
