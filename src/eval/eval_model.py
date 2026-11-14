"""评测：学生模型 vs 规则 oracle。

指标说明（清洗任务的特殊情况：输出必须"只删不增"，所以不能用普通生成指标）：

* ``exact``        与标准答案逐字符一致 —— 最严格，只有完全对齐才算对；
* ``sim``          字符级相似度（SequenceMatcher），容忍一处标点差异；
* ``loss``         标准答案里有、模型输出里没有的字符占比 —— 漏删/漏改；
* ``add``          模型输出里有、标准答案里没有的字符占比 —— 改写/幻觉；
* ``len_ratio``    输出/答案长度比，异常值通常意味着模型开始"解释"。

用法：
    python src/eval/eval_model.py --model release/Qwen3.5-0.8B-TextPreprocessor \
        --data data/rule/test.jsonl --out eval/rule_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.format import to_prompt_messages  # noqa: E402


def char_metrics(pred: str, ref: str) -> tuple[float, float, float]:
    """返回 (缺陷率 loss, 新增率 add, 相似度 sim)。"""
    if not ref:
        return (0.0, 0.0, 1.0 if not pred else 0.0)
    matcher = SequenceMatcher(None, ref, pred, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    loss = max(0.0, (len(ref) - matched) / len(ref))
    add = max(0.0, (len(pred) - matched) / max(1, len(ref)))
    sim = matcher.ratio()
    return loss, add, sim


def score(preds: list[str], refs: list[str], raws: list[str]) -> dict:
    n = len(refs)
    exact = 0
    sims: list[float] = []
    losses: list[float] = []
    adds: list[float] = []
    lengths: list[float] = []
    bad = 0
    for pred, ref, raw in zip(preds, refs, raws):
        pred = (pred or "").strip()
        if pred == ref.strip():
            exact += 1
        loss, add, sim = char_metrics(pred, ref.strip())
        sims.append(sim)
        losses.append(loss)
        adds.append(add)
        lengths.append(len(pred) / max(1, len(ref.strip())))
        # 明显跑偏：输出长度相对标准答案超过 ±30%（模型开始解释/截断）
        if len(pred) > 1.3 * max(1, len(ref.strip())) or len(pred) < 0.7 * max(1, len(ref.strip())):
            bad += 1
    return {
        "n": n,
        "exact": round(exact / max(1, n), 4),
        "sim": round(sum(sims) / max(1, n), 4),
        "loss": round(sum(losses) / max(1, n), 4),
        "add": round(sum(adds) / max(1, n), 4),
        "len_ratio": round(sum(lengths) / max(1, n), 3),
        "abnormal": bad,
    }


def format_table(rows: dict[str, dict]) -> str:
    head = "| 方案 | 样本数 | 完全一致 | 相似度 | 漏删率 | 新增率 | 长度比 | 异常输出 |\n"
    head += "|---|---|---|---|---|---|---|---|\n"
    body = ""
    for name, m in rows.items():
        body += (
            f"| {name} | {m['n']} | {m['exact']:.1%} | {m['sim']:.3f} | {m['loss']:.3f} "
            f"| {m['add']:.3f} | {m['len_ratio']:.2f} | {m['abnormal']} |\n"
        )
    return head + body


def load_dataset(path: Path, limit: int = 0) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                msgs = obj["messages"]
                raw = next(m["content"] for m in msgs if m["role"] == "user")
                ref = next(m["content"] for m in msgs if m["role"] == "assistant")
            else:
                raw, ref = obj["raw"], obj["clean"]
            rows.append({"raw": raw, "ref": ref})
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True, help="可多次指定，如 rule/test.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--merge-adapter", default="", help="可选 LoRA 适配器目录")
    ap.add_argument("--baseline", action="store_true", help="同时跑规则 oracle 作为基线")
    ap.add_argument("--save-preds", default="", help="把逐条预测存成 JSONL，便于人工抽查")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, trust_remote_code=True, device_map="auto"
    )
    if args.merge_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.merge_adapter)
    model.eval()

    all_rows = []
    for spec in args.data:
        path = Path(spec)
        rows = load_dataset(path, args.limit)
        print(f"{path}: {len(rows)} 条")
        all_rows.append((f"{path.parent.name}/{path.stem}", rows))

    results = {}
    for name, rows in all_rows:
        preds: list[str] = []
        start = time.time()
        for row in rows:
            text = tokenizer.apply_chat_template(
                to_prompt_messages(row["raw"]), tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=min(args.max_new_tokens, max(64, int(len(row["raw"]) * 1.4) + 64)),
                    do_sample=False,
                    num_beams=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = out[0][inputs["input_ids"].shape[1] :]
            preds.append(tokenizer.decode(generated, skip_special_tokens=True))
        elapsed = time.time() - start
        refs = [r["ref"] for r in rows]
        raws = [r["raw"] for r in rows]
        model_score = score(preds, refs, raws)
        model_score["sec_per_sample"] = round(elapsed / max(1, len(rows)), 3)
        results[name] = model_score
        if args.save_preds:
            safe_name = name.replace("/", "_").replace("\\", "_")
            out_pred = Path(args.save_preds).with_name(f"{Path(args.save_preds).stem}_{safe_name}.jsonl")
            out_pred.parent.mkdir(parents=True, exist_ok=True)
            with out_pred.open("w", encoding="utf-8") as fh:
                for row, pred in zip(rows, preds):
                    fh.write(json.dumps(
                        {"raw": row["raw"], "ref": row["ref"], "pred": pred},
                        ensure_ascii=False) + "\n")
            print(f"  逐条预测 -> {out_pred}")
        print(f"  [{name}] exact={model_score['exact']:.1%} sim={model_score['sim']:.3f} "
              f"{model_score['sec_per_sample']}s/条")

    print()
    print(format_table(results))
    if args.baseline:
        from cleaner.normalizer import clean

        base_results = {}
        for name, rows in all_rows:
            preds = [clean(r["raw"]) for r in rows]
            base_results[name] = score(preds, [r["ref"] for r in rows], [r["raw"] for r in rows])
            base_results[name]["sec_per_sample"] = 0.0
            print(f"  [baseline {name}] exact={base_results[name]['exact']:.1%} "
                  f"sim={base_results[name]['sim']:.3f}")
        print()
        print(format_table({f"规则基线 · {k}": v for k, v in base_results.items()}))
        results.update({f"rule_baseline_{k}": v for k, v in base_results.items()})
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"指标已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
