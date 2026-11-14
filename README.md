# 输入文本预处理模型（Qwen3.5-0.8B · 蒸馏自 DeepSeek-V4.1-Flash）

把用户随手粘贴的**脏文本**清洗成可直接喂给下游大模型的**干净文本**：删不可见字符、去 HTML、
统一全角半角与中文标点、压缩无意义重复、整理空白与断行、判断"这段本来就干净"。

> 约束：只改格式，不改内容。不改写、不纠错、不翻译、不总结、不脱敏。

| 项目 | 值 |
|---|---|
| 学生模型 | `Qwen/Qwen3.5-0.8B`（Apache-2.0，原生多模态结构，本任务只用文本通道） |
| 老师模型 | `deepseek-flash` = DeepSeek-V4.1-Flash（1M 上下文） |
| 训练方式 | LoRA（r=16）SFT，标签由规则 oracle 精确生成 + 老师数据交叉校验 |
| 参数量 | 0.85B（含视觉塔），LoRA 可训练 7.27M（0.85%） |
| 输出语言 | 与输入一致 |

## 效果

| 测试集 | 样本数 | 完全一致 | 字符相似度 | 漏删率 | 新增率 |
|---|---|---|---|---|---|
| 规则合成测试集 | 600 | **91.5%** | 0.994 | 0.7% | 0.4% |
| 老师生成测试集（真实场景） | 148 | **86.5%** | 0.997 | 0.3% | 0.3% |
| 规则基线（同一规范的可执行实现） | — | 100% | 1.000 | 0 | 0 |

规则基线是标签的来源，所以它 100% 是定义使然；学生与基线的差距就是蒸馏损失。
**新增率 0.3-0.4% 是更值得看的数字**——它衡量模型有没有偷偷改写内容，越低越好。

失败案例见 [examples.md](examples.md)。

## 目录结构

```
.
├── Qwen3.5-0.8B-TextPreprocessor/   蒸馏好的模型（可直接 from_pretrained）
├── docs/
│   ├── TASK_SPEC.md            任务规范（数据、规则、评测三方契约）
│   ├── DATA_DESIGN.md          数据构造思路与踩坑
│   ├── TRAINING.md             8GB 显卡上的训练实录
│   └── EVALUATION.md           指标设计与读法
├── src/
│   ├── cleaner/normalizer.py   规则引擎（也是训练标签的 oracle）
│   ├── data/                   种子语料、噪声注入、老师调用、SFT 组装
│   ├── train/                  LoRA 训练与权重合并
│   ├── eval/eval_model.py      评测（含规则基线对照）
│   └── serve/infer.py          命令行推理
├── tests/test_normalizer.py    规范契约测试（35 条）
├── tools/                      可逆性自检、成品自检、模型卡与示例生成
├── dataset/                    公开的评测子集
└── release/                    发布清单与上传脚本
```

## 快速开始

```bash
pip install torch transformers peft accelerate
python src/serve/infer.py --model Qwen3.5-0.8B-TextPreprocessor \
    --text "  我我我  明天   去北京！！！  "
```

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer

SYSTEM = ("你是输入文本预处理器。对用户给出的原始文本做清洗规范化：删除不可见字符与噪声、"
          "去掉 HTML 标签、统一全角半角与标点、压缩无意义重复、整理空白与断行。"
          "只改格式不改内容，直接输出清洗后的文本，不要解释。")

tok = AutoTokenizer.from_pretrained("Qwen3.5-0.8B-TextPreprocessor")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen3.5-0.8B-TextPreprocessor", dtype="float16", device_map="auto")

msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": raw}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

> **乱码先修**：`æ˜Žå¤©` 这类字节级损坏请在调用模型前用 `repair_mojibake()` 处理，
> 理由见 `docs/TRAINING.md`。模型本身不负责乱码。

## 复现训练

```powershell
# 0) 建环境（Python 3.10 + CUDA 12.8）
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install transformers peft accelerate datasets safetensors

# 1) 下载学生模型（也可以 huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B）

# 2) 合成规则数据
python src/data/build_rule_pairs.py --train 12000 --valid 600 --test 600

# 3) 让老师造自然分布的样本（需要 DEEPSEEK_API_KEY）
$env:DEEPSEEK_API_KEY = "sk-..."
python src/data/teacher_pairs.py --mode gen --n 1200 --workers 10
python tools/relabel_teacher.py --in data/teacher/gen.jsonl --out data/teacher/gen_v2.jsonl

# 4) 合并训练集
python src/data/build_sft.py --teacher data/teacher/gen_v2.jsonl --teacher-repeat 2

# 5) 训练 + 合并
python src/train/sft_lora.py --data data/sft/train.jsonl --valid data/sft/valid.jsonl `
    --out outputs/lora --epochs 2 --batch-size 1 --grad-accum 16 --max-len 512
python src/train/merge_lora.py --base models/Qwen3.5-0.8B --adapter outputs/lora `
    --out Qwen3.5-0.8B-TextPreprocessor

# 6) 评测 + 模型卡
python src/eval/eval_model.py --model Qwen3.5-0.8B-TextPreprocessor `
    --data data/rule/test.jsonl --baseline --out eval/rule_test.json
```

## 数据是怎么造的

**规则合成（主力）**：原创种子语料 → 11 种噪声注入器弄脏 → 规则引擎清洗得到标签。
注入器与清洗步骤一一对应，`tools/check_invertibility.py` 验证 11 种噪声的
"清洗后完全还原"比例均为 100%，标签是算出来的、可复现、零标注成本。

**老师生成（补充）**：DeepSeek-V4.1-Flash 按 16 种真实场景造出"真实会遇到的脏输入"，
其清洗结果必须与规则 oracle 相似度 ≥0.90 才被采用，标签最终仍以规范为准。
这条校验挡住的不只是格式不一致，还有幻觉——老师会"猜"出被截断的乱码原文，那不是修复。

细节见 [docs/DATA_DESIGN.md](docs/DATA_DESIGN.md)。

## 已知限制

- **不修乱码**：小模型学不会字节重排，硬教会让它凭空编造内容（实测过）。
  请在调用前用 `repair_mojibake()`。规则引擎 `clean()` 已内置这一步。
- 不纠错别字：`不不会会` 保持原样——两字叠词是合法词，误伤代价太高。
- 不脱敏：手机号、邮箱、身份证号原样保留。
- 不翻译、不繁简转换。
- 表格的列对齐会被空格压缩破坏。
- 输入超过 ~3000 字时建议分段调用。

详见 [docs/TASK_SPEC.md](docs/TASK_SPEC.md)。

## 许可证

代码与权重：Apache-2.0。底座 `Qwen3.5-0.8B` 为 Apache-2.0；老师模型的权重未参与分发，
本项目仅使用其生成的文本作为训练目标。
