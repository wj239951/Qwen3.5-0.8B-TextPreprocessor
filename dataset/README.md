# 输入文本清洗数据集（评测子集）

本数据集是「输入文本清洗规范化」任务的**评测子集**，随
[Qwen3.5-0.8B-TextPreprocessor](../release/Qwen3.5-0.8B-TextPreprocessor) 一起发布，
用于复现模型卡里的评测数字，或对比你自己的方案。

## 文件

| 文件 | 条数 | 说明 |
|---|---|---|
| `rule_test.jsonl` | 600 | 由原创种子语料注入 11 类噪声后用规则 oracle 打标 |
| `teacher_test.jsonl` | 153 | DeepSeek-V4.1-Flash 生成的"真实场景脏文本"，标签经规则 oracle 交叉校验 |
| `train_sample.jsonl` | 200 | 训练集随机抽样，仅供了解分布 |

每行字段：

```json
{"raw": "脏文本", "clean": "清洗后的标准答案", "src": "rule|teacher"}
```

`teacher_test.jsonl` 额外有 `teacher_clean`（老师原始答案）与 `similarity`（与 oracle 的相似度），
可以用来观察"老师与规范的分歧点"，这些点恰好是任务里最模糊的地方。

## 任务定义

只改格式，不改内容：删不可见字符 / 修乱码 / 去 HTML / 全角半角与标点归一 /
压缩 ≥3 次的重复 / 整理空白与断行。**禁止**改写、纠错、翻译、总结、脱敏。

完整规则见 [docs/TASK_SPEC.md](../docs/TASK_SPEC.md)。

## 评测方式

```bash
python src/eval/eval_model.py --model <你的模型> --data dataset/rule_test.jsonl --baseline
```

主要指标：

- **exact**：与标准答案逐字符一致的比例（最严格）；
- **sim**：字符级相似度；
- **loss / add**：漏删率 / 新增率（判断有没有偷偷改写内容）。

## 噪声类型

`zero_width`、`control_chars`、`fullwidth`、`halfwidth_punct`、`char_repeat`、
`punct_repeat`、`messy_space`、`blank_lines`、`html_wrap`、`mojibake`、`hard_wrap`。

每个注入器都通过"清洗后完全还原"的自检（见 `tools/check_invertibility.py`），
因此标签是算出来的，不是标出来的。

## 许可

CC BY 4.0。种子语料为本项目原创，老师数据由模型生成。
