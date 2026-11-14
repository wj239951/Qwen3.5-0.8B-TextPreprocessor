"""把 LoRA 适配器合并回底座，产出可直接分发的完整模型目录。

合并后的目录是自包含的：config、权重、tokenizer、chat template、generation config
全部齐备，``AutoModelForImageTextToText.from_pretrained(dir)`` 即可加载。

用法：
    python src/train/merge_lora.py --base models/Qwen3.5-0.8B --adapter outputs/lora/best \
        --out release/Qwen3.5-0.8B-TextPreprocessor
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402

#: 需要跟着权重一起复制的辅助文件
COPY_FILES = (
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "generation_config.json",
)


def load_base(path: str):
    kwargs = {"trust_remote_code": True}
    for key in ("dtype", "torch_dtype"):
        try:
            return AutoModelForImageTextToText.from_pretrained(path, **{key: torch.float16}, **kwargs)
        except TypeError as exc:
            if "unexpected keyword" in str(exc):
                continue
            raise
    return AutoModelForImageTextToText.from_pretrained(path, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shard-size", default="4GB")
    args = ap.parse_args()

    base_path = Path(args.base)
    adapter_path = Path(args.adapter)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"加载底座 {base_path}")
    model = load_base(str(base_path))
    print(f"加载适配器 {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()
    model = model.half()
    model.config.use_cache = True

    print(f"保存到 {out}")
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size=args.max_shard_size)

    tokenizer = AutoTokenizer.from_pretrained(str(base_path), trust_remote_code=True)
    # chat template 必须显式写进 tokenizer_config.json：
    # 只靠同目录的 chat_template.jinja 在某些加载路径下会丢，用户 apply_chat_template 就报错
    template_file = base_path / "chat_template.jinja"
    if template_file.exists() and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = template_file.read_text(encoding="utf-8")
    tokenizer.save_pretrained(str(out))

    # 生成配置要跟 tokenizer 对齐：底座的 config 里 eos 是 <|endoftext|>，
    # 但对话模板实际用 <|im_end|> 收尾。不修的话用户 model.generate() 会停不下来。
    gen = model.generation_config
    gen.eos_token_id = tokenizer.eos_token_id
    gen.pad_token_id = tokenizer.pad_token_id
    gen._from_model_config = False
    gen.save_pretrained(str(out))

    for name in COPY_FILES:
        src = base_path / name
        if src.exists():
            shutil.copy2(src, out / name)
    for name in ("LICENSE", "NOTICE"):
        src = base_path / name
        if src.exists():
            shutil.copy2(src, out / name)

    # 留一份合并来源记录，方便复现
    meta = {
        "base_model": str(base_path),
        "adapter": str(adapter_path),
        "merged_with": "peft.merge_and_unload",
        "dtype": "float16",
    }
    (out / "merge_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"完成，共 {total / 1024 ** 3:.2f} GB，{len(list(out.iterdir()))} 个条目")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
