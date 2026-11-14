"""发布前自检：像用户一样加载 release 目录，跑一批用例。

验证的是"交付物本身"是否可用：目录是否自包含、tokenizer/chat template 是否随权重一起走了、
生成是否正常、是否真的会清洗。任何一项失败都不应该发布。

用法：
    python tools/verify_release.py --model release/Qwen3.5-0.8B-TextPreprocessor --limit 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.format import to_prompt_messages  # noqa: E402

#: 必须通过的基础用例（输入, 期望）。期望值全部由 docs/TASK_SPEC.md 推导，
#: 并已与规则 oracle 的输出逐字对齐——不要凭直觉写"更漂亮"的答案。
#: 注意：模型不负责乱码（那属于字节层还原，见 repair_mojibake），所以这里没有乱码用例。
GOLDEN = [
    ("  我我我  明天   去北京！！！  ", "我 明天 去北京！"),
    ("<p>你好&nbsp;世界</p><p>第二段</p>", "你好 世界\n\n第二段"),
    ("Ｈｅｌｌｏ　１２３", "Hello 123"),
    # 反向用例：本来就干净、且含合法两字叠词的文本，一个字都不能改
    ("订单号 1000000，刚刚发货，谢谢", "订单号 1000000，刚刚发货，谢谢"),
    ("运行 `pip install foo` 即可", "运行 `pip install foo` 即可"),
    ("妈妈说要早点回家，慢慢来", "妈妈说要早点回家，慢慢来"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=12, help="额外从测试集抽多少条")
    ap.add_argument("--data", default=str(ROOT / "data" / "rule" / "test.jsonl"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    model_dir = Path(args.model)
    files = sorted(p.name for p in model_dir.iterdir() if p.is_file())
    print(f"目录 {model_dir}，{len(files)} 个文件")
    for must in ("config.json", "tokenizer_config.json", "chat_template.jinja"):
        if must not in files:
            print(f"缺少必要文件: {must}")
            return 1

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    ).eval()
    print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B，"
          f"显存 {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB" if torch.cuda.is_available() else "")

    def run(raw: str) -> str:
        prompt = tok.apply_chat_template(to_prompt_messages(raw), tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max(64, int(len(raw) * 1.4) + 48),
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    ok = 0
    print("\n=== 基础用例 ===")
    for raw, expected in GOLDEN:
        got = run(raw)
        flag = "PASS" if got == expected else "FAIL"
        ok += flag == "PASS"
        print(f"{flag}  {raw!r}")
        if got != expected:
            print(f"      期望 {expected!r}")
            print(f"      实际 {got!r}")
    print(f"基础用例 {ok}/{len(GOLDEN)}")

    if args.limit:
        data = Path(args.data)
        rows = []
        with data.open(encoding="utf-8") as fh:
            for line in fh:
                rows.append(json.loads(line))
                if len(rows) >= args.limit:
                    break
        hits = 0
        print(f"\n=== 测试集抽样 {len(rows)} 条 ===")
        for row in rows:
            got = run(row["raw"])
            hits += got == row["clean"]
            if got != row["clean"]:
                print(f"  raw:  {row['raw'][:60]!r}")
                print(f"  want: {row['clean'][:60]!r}")
                print(f"  got:  {got[:60]!r}")
        print(f"抽样命中 {hits}/{len(rows)}")

    return 0 if ok == len(GOLDEN) else 1


if __name__ == "__main__":
    raise SystemExit(main())
