"""命令行推理：单条、文件、管道三种用法。

    python src/serve/infer.py --model release/Qwen3.5-0.8B-TextPreprocessor --text "  你好\u200b世界  "
    python src/serve/infer.py --model ... --file dirty.txt
    type dirty.txt | python src/serve/infer.py --model ... --stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.format import to_prompt_messages  # noqa: E402


def build(model_path: str, adapter: str | None = None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def clean_text(tokenizer, model, raw: str, max_new_tokens: int = 1024) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        to_prompt_messages(raw), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    budget = min(max_new_tokens, max(64, int(len(raw) * 1.4) + 64))
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=budget,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--text", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    tokenizer, model = build(args.model, args.adapter or None)

    if args.text:
        print(clean_text(tokenizer, model, args.text, args.max_new_tokens))
    elif args.file:
        raw = Path(args.file).read_text(encoding="utf-8")
        print(clean_text(tokenizer, model, raw, args.max_new_tokens))
    elif args.stdin:
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line:
                print(clean_text(tokenizer, model, line, args.max_new_tokens))
    else:
        print("交互模式，空行退出：")
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip():
                break
            print(clean_text(tokenizer, model, line, args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
