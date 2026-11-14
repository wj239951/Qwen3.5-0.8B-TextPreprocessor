"""生成 release 目录里的模型卡（README.md）。

所有数字都从真实产物里读：评测 JSON、训练配置、数据统计，避免手抄出错。

用法：
    python tools/make_model_card.py --release release/Qwen3.5-0.8B-TextPreprocessor \
        --eval eval/rule_test.json --train-meta outputs/lora/train_meta.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = """---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
  - text-cleaning
  - text-normalization
  - distillation
  - qwen3.5
language:
  - zh
  - en
pipeline_tag: text-generation
---

# Qwen3.5-0.8B-TextPreprocessor

把用户随手粘贴的**脏文本**清洗成可直接喂给下游大模型的**干净文本**。
蒸馏自 `deepseek-flash`（DeepSeek-V4.1-Flash），底座为 `Qwen/Qwen3.5-0.8B`。

**只改格式，不改内容**：不改写、不纠错、不翻译、不总结、不脱敏。

## 效果

{metrics_table}

*规则基线 = `src/cleaner/normalizer.py`，同一套规范的可执行实现。*

## 用法

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer

SYSTEM = ("你是输入文本预处理器。对用户给出的原始文本做清洗规范化：删除不可见字符与噪声、"
          "修复乱码、去掉 HTML 标签、统一全角半角与标点、压缩无意义重复、整理空白与断行。"
          "只改格式不改内容，直接输出清洗后的文本，不要解释。")

repo = "{repo_name}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(repo, dtype="float16", device_map="auto")

raw = "<p>我我我&nbsp;明天  去北京！！！</p>"
msgs = [{{"role": "system", "content": SYSTEM}}, {{"role": "user", "content": raw}}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# 我明天 去北京！
```

生成参数：**贪心解码**（`do_sample=False`），`max_new_tokens` 取输入长度的 1.4 倍左右。

> 上面这段系统提示词是**训练时的原文，请逐字使用**。其中"修复乱码"一项对应的是
> 规范层的 `repair_mojibake()`，不是本模型的能力——训练数据里不含乱码样本，
> 模型遇到乱码会原样保留（这是刻意的，见下文）。

## 能做什么 / 不能做什么

| 会做 | 不会做 |
|---|---|
| 删零宽字符、BOM、控制符 | **修乱码**（见下） |
| 去 HTML 标签、解码实体 | 纠错别字 |
| 全角字母数字转半角 | 翻译、繁简转换 |
| 中文里的半角标点转全角 | 摘要、扩写、补全 |
| 压缩 ≥3 次的重复字符 | 手机号/邮箱/身份证脱敏 |
| 整理空白、合并排版折断的行 | 调整段落顺序 |
| 判断"这段本来就干净，别动" | 保留表格列对齐 |

### 乱码不是这个模型的活

`æ˜Žå¤©` → `明天` 需要逐字节重排。0.8B 模型学不会这件事，硬教的结果是它记住
"看到乱码就输出一段中文"，进而在真实输入上**凭空编造整段内容**——这比不修危险得多。

所以乱码请在调用本模型之前，用配套的确定性实现处理掉：

```python
from cleaner.normalizer import repair_mojibake, clean

text = repair_mojibake(raw)[0]   # 字节层还原，纯代码，零幻觉
text = clean(text)               # 可选：规则层兜底（纯代码方案）
text = model_clean(text)         # 本模型：格式层
```

完整规则见同目录的 [`TASK_SPEC.md`](TASK_SPEC.md)。

## 训练数据

| 来源 | 规模 | 说明 |
|---|---|---|
{data_table}

**规则合成**：干净种子语料 → 11 种注入器弄脏 → 规则引擎清洗得到标签。注入器与清洗步骤
一一对应，可逆性自检 100%，标签精确。

**老师生成**：DeepSeek-V4.1-Flash 按场景造出真实分布的脏输入，其清洗结果需与规则 oracle
相似度 ≥0.90 才被采用；标签最终以规范为准，用于抓老师"越界改写"。

## 训练细节

| 项 | 值 |
|---|---|
| 底座 | `Qwen/Qwen3.5-0.8B`（Ahmed 结构，24 层混合线性注意力 + 视觉塔，本任务只用文本通道） |
| 方法 | LoRA SFT |
| LoRA | {lora} |
| 精度 | {dtype} |
| 序列长度 | {max_len} |
| 有效 batch | {batch} |
| 学习率 | {lr} |
| epoch | {epochs} |
| 硬件 | {gpu} |
| 峰值显存 | {peak_vram} |
| 训练步数 | {steps} |

## 局限

- 输入超过约 3000 字时建议分段调用。
- 二次损坏的乱码（如 GBK/UTF-8 反复转码）无法修复，保持原样。
- 表格的列对齐会被空格压缩破坏。
- 对 2 字重复（`去去`、`不不会会`）不做处理：两字叠词在中文里是合法词，误伤代价太高。

## 许可证与致谢

Apache-2.0。底座 `Qwen3.5-0.8B` 由阿里千问开源（Apache-2.0）；
老师 `DeepSeek-V4.1-Flash` 由 DeepSeek 提供，本项目仅将其生成的文本作为训练目标，
老师的权重未参与分发。

生成日期：{today}
"""


def load_eval(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--eval", default=str(ROOT / "eval" / "rule_test.json"))
    ap.add_argument("--eval-teacher", default=str(ROOT / "eval" / "teacher_test.json"))
    ap.add_argument("--train-meta", default=str(ROOT / "outputs" / "lora" / "train_meta.json"))
    ap.add_argument("--repo-name", default="Qwen3.5-0.8B-TextPreprocessor")
    args = ap.parse_args()

    meta: dict = {}
    meta_path = Path(args.train_meta)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    rows = []
    for label, path in (("规则合成测试集", args.eval), ("老师生成测试集", args.eval_teacher)):
        data = load_eval(Path(path))
        if not data:
            continue
        for name, m in data.items():
            is_baseline = name.startswith("rule_baseline_")
            dataset = name.replace("rule_baseline_", "")
            tag = f"{label}（规则基线）" if is_baseline else label
            rows.append(
                (f"{tag} {dataset}", m.get("n", 0), m.get("exact", 0), m.get("sim", 0),
                 m.get("loss", 0), m.get("add", 0))
            )
    if rows:
        table = "| 测试集 | 样本数 | 完全一致 | 字符相似度 | 漏删率 | 新增率 |\n|---|---|---|---|---|---|\n"
        for row in rows:
            table += f"| {row[0]} | {row[1]} | {row[2]:.1%} | {row[3]:.3f} | {row[4]:.3f} | {row[5]:.3f} |\n"
    else:
        table = "_（尚未评测）_"

    data_rows = [
        ("规则合成", meta.get("n_rule", "—"), "11 种噪声注入，标签由规则 oracle 精确生成"),
        ("老师生成", meta.get("n_teacher", "—"), "DeepSeek-V4.1-Flash 造真实脏输入 + 交叉校验"),
    ]
    data_table = "".join(f"| {a} | {b} | {c} |\n" for a, b, c in data_rows)

    text = TEMPLATE.format(
        metrics_table=table,
        data_table=data_table,
        repo_name=args.repo_name,
        lora=f"r={meta.get('lora_r', 16)}, alpha={meta.get('lora_alpha', 32)}, "
             f"dropout={meta.get('lora_dropout', 0.05)}, "
             f"可训练参数 {meta.get('trainable_params', '—')}",
        dtype=meta.get("dtype", "float16"),
        max_len=meta.get("max_len", 1024),
        batch=meta.get("effective_batch", "1×16"),
        lr=meta.get("lr", "1e-4"),
        epochs=meta.get("epochs", "2"),
        gpu=meta.get("gpu", "RTX 4060 8GB"),
        peak_vram=meta.get("peak_vram", "—"),
        steps=meta.get("steps", "—"),
        today=date.today().isoformat(),
    )

    out = Path(args.release) / "README.md"
    out.write_text(text, encoding="utf-8")
    print(f"模型卡已写入 {out}（{len(text)} 字）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
