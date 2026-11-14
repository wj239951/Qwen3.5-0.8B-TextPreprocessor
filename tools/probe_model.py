"""一次性探针：确认本机能加载 Qwen3.5-0.8B、跑通 chat template、前向与生成。

训练前必跑，用来尽早暴露架构支持、显存、模板问题。

用法：
    python tools/probe_model.py --model models/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--generate", type=int, default=32)
    args = ap.parse_args()

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"tokenizer 加载 {time.time() - t0:.1f}s, vocab={tok.vocab_size}, "
          f"eos={tok.eos_token!r}, pad={tok.pad_token!r}")

    msgs = [
        {"role": "system", "content": "你是输入文本预处理器。只改格式不改内容。"},
        {"role": "user", "content": "  我我我  明天  去北京！！！  "},
    ]
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": "我明天 去北京！"}],
                                   tokenize=False, add_generation_prompt=False)
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"prompt 尾部: {prefix[-40:]!r}")
    ids = tok(full, add_special_tokens=False)["input_ids"]
    pids = tok(prefix, add_special_tokens=False)["input_ids"]
    print(f"full={len(ids)} tokens, prefix={len(pids)} tokens, prefix 是前缀: {ids[:len(pids)] == pids}")

    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True, device_map={"": 0}
    )
    model.eval()
    print(f"模型加载 {time.time() - t0:.1f}s, 参数量 "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.3f}B, "
          f"显存 {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB")
    print("顶层模块:", [n for n, _ in model.named_children()][:8])

    batch = tok(prefix, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        logits = model(**batch).logits
    print(f"前向 logits {tuple(logits.shape)}")

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=args.generate, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    dt = time.time() - t0
    text = tok.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"生成 {args.generate} tokens 用时 {dt:.1f}s ({args.generate / dt:.1f} tok/s)")
    print(f"未微调输出: {text!r}")
    print(f"峰值显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
