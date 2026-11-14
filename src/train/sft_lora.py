"""Qwen3.5-0.8B 的 LoRA 蒸馏训练。

学生模型是原生多模态结构（``Qwen3_5ForConditionalGeneration``），本任务只用文本通道，
因此直接走文本 forward，不喂 pixel_values。

显存只有 8GB，策略：
* LoRA（r=16）+ 冻结底座；
* 梯度检查点；
* batch=1 + 梯度累积；
* fp16/bf16 自适应。

用法：
    python src/train/sft_lora.py --data data/sft/train.jsonl --valid data/sft/valid.jsonl \
        --out outputs/lora --epochs 2
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import transformers  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

TEXT_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj", "out_proj", "qkv_proj",
)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #

class CleanSFTDataset(Dataset):
    """把 messages 编码成 (input_ids, labels)，只对 assistant 段计算 loss。"""

    def __init__(self, path: str, tokenizer, max_len: int, limit: int | None = None) -> None:
        self.rows: list[dict] = []
        self.skipped = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                item = self._encode(rec["messages"], tokenizer, max_len)
                if item is None:
                    self.skipped += 1
                    continue
                self.rows.append(item)
                if limit and len(self.rows) >= limit:
                    break

    @staticmethod
    def _encode(messages: list[dict], tokenizer, max_len: int) -> dict | None:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prefix_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        if len(ids) > max_len:
            return None
        # 正常情况下 prefix 一定严格是前缀；个别模板差异时退化为按长度对齐
        if ids[: len(prefix_ids)] != prefix_ids:
            prefix_len = len(prefix_ids)
        else:
            prefix_len = len(prefix_ids)
        labels = [-100] * prefix_len + ids[prefix_len:]
        if all(label == -100 for label in labels):
            return None
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class PadCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #

def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model(model_path: str):
    """兼容 transformers 4.x / 5.x 的 dtype 参数命名。"""
    dtype = pick_dtype()
    kwargs = {"trust_remote_code": True, "device_map": {"": 0} if torch.cuda.is_available() else None}
    for key in ("dtype", "torch_dtype"):
        try:
            return AutoModelForImageTextToText.from_pretrained(model_path, **{key: dtype}, **kwargs)
        except TypeError as exc:
            if "unexpected keyword" in str(exc):
                continue
            raise
    return AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)


def find_text_targets(model) -> list[str]:
    """只给文本主干里的线性层挂 LoRA，视觉塔保持冻结。"""
    names: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "vision" in name or "visual" in name or "lm_head" in name:
            continue
        if any(name.endswith(suffix) for suffix in TEXT_TARGET_SUFFIXES):
            names.append(name)
    # 去掉多余前缀，peft 接受后缀即可
    suffixes = sorted({name.rsplit(".", 1)[-1] for name in names})
    return suffixes


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #

class MemWatch(TrainerCallback):
    """每 N 步打印一次显存，用来发现缓慢增长的泄漏或碎片。"""

    def __init__(self, every: int = 20) -> None:
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if state.global_step % self.every == 0 and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            free, total = torch.cuda.mem_get_info()
            print(f"[mem] step={state.global_step} alloc={alloc:.2f}GiB "
                  f"reserved={reserved:.2f}GiB system_free={free / 1024 ** 3:.2f}/{total / 1024 ** 3:.2f}GiB")
        return control


def make_args(**kwargs) -> TrainingArguments:
    """屏蔽 transformers 版本间的参数改名（evaluation_strategy / eval_strategy 等）。"""
    params = inspect.signature(TrainingArguments.__init__).parameters
    for old, new in (("evaluation_strategy", "eval_strategy"), ("save_safetensors", "save_safetensors")):
        if old in kwargs and old not in params and new in params:
            kwargs[new] = kwargs.pop(old)
    kwargs = {k: v for k, v in kwargs.items() if k in params or k == "self"}
    return TrainingArguments(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--data", default=str(ROOT / "data" / "sft" / "train.jsonl"))
    ap.add_argument("--valid", default=str(ROOT / "data" / "sft" / "valid.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "lora"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--save-steps", type=int, default=400)
    ap.add_argument("--no-grad-ckpt", action="store_true", help="关闭梯度检查点（更快，但更吃显存）")
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="显存硬上限占比（0 表示不限制）。设了之后超限直接 OOM，而不是走共享内存假死")
    args = ap.parse_args()

    print(f"torch={torch.__version__} transformers={transformers.__version__} "
          f"cuda={torch.cuda.is_available()} dtype={pick_dtype()}")
    if args.mem_fraction and torch.cuda.is_available():
        # Windows 的 WDDM 在显存超限时会退到共享内存，表现为训练"假死"。
        # 设一个硬上限，让超限变成显式 OOM，问题立刻暴露。
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device=0)
        print(f"显存上限设为 {args.mem_fraction:.0%}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = CleanSFTDataset(args.data, tokenizer, args.max_len, limit=args.limit or None)
    valid_ds = CleanSFTDataset(args.valid, tokenizer, args.max_len, limit=200)
    print(f"训练样本 {len(train_ds)} 条（超长跳过 {train_ds.skipped}），验证样本 {len(valid_ds)} 条")
    if not len(train_ds):
        return 1

    model = load_model(args.model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable") and not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    targets = find_text_targets(model)
    print(f"LoRA target modules: {targets}")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = make_args(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        bf16=pick_dtype() is torch.bfloat16,
        fp16=pick_dtype() is torch.float16,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        max_steps=args.max_steps,
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=2026,
        logging_first_step=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        callbacks=[MemWatch(every=args.logging_steps)],
    )

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    trainer.train()
    if torch.cuda.is_available():
        print(f"峰值显存 {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    # 训练元信息：模型卡里的数字全部来自这里，避免手抄
    src_counter: dict[str, int] = {}
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                src = json.loads(line).get("src", "unknown")
                src_counter[src] = src_counter.get(src, 0) + 1
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    meta = {
        "base_model": args.model,
        "n_train": len(train_ds),
        "n_valid": len(valid_ds),
        "n_rule": src_counter.get("rule", 0),
        "n_teacher": src_counter.get("teacher", 0),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_params": f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M",
        "dtype": str(pick_dtype()).replace("torch.", ""),
        "max_len": args.max_len,
        "effective_batch": f"{args.batch_size}x{args.grad_accum}",
        "lr": args.lr,
        "epochs": args.epochs,
        "steps": trainer.state.global_step,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_vram": f"{peak:.2f} GiB",
    }
    (Path(args.out) / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"LoRA 已保存到 {args.out}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
