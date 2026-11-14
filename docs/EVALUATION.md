# 评测说明

## 为什么这个任务的评测不能只看 BLEU / ROUGE

清洗任务的输出**必须能逐字回填**。生成指标（BLEU、ROUGE）对"多删了一个空格"和
"把'谢谢'改成'谢'"给出的分数可能差不多，但后者是内容损伤。所以主指标设计成

- **exact**：与标准答案逐字符一致。最严格，也最贴近真实使用。
- **sim**：字符级相似度（`difflib.SequenceMatcher`）。用来区分"差一个标点"和"全错"。
- **loss**：标准答案里有、模型输出里没有的字符占比 → **漏删/漏改**。
- **add**：模型输出里有、标准答案里没有的字符占比 → **改写 / 幻觉**。
- **len_ratio**：输出长度 / 答案长度。异常值通常意味着模型开始解释。
- **abnormal**：长度偏离答案 ±30% 的样本数。

`add` 是最需要盯的指标。一个只会删字符的模型，`add` 应该接近 0；
`add` 升高说明模型在"帮忙"——补字、改病句、翻译，都是这个任务明令禁止的。

## 测试集

| 集合 | 条数 | 用途 |
|---|---|---|
| `data/rule/test.jsonl` | 600 | 规则合成，覆盖面广，标签由 oracle 精确生成 |
| `data/teacher/test_v2.jsonl` | 153 | 老师造的真实场景脏文本，标签经 oracle 校验 |

两个集合都按**种子**与训练集隔离，同一篇种子的变体不会跨集合。

因为标签就是 oracle 的输出，规则基线在这两个集合上是 **100%**。这不是作弊，
而是刻意的设计：评测衡量的是"学生多接近规范"，基线给出上限，学生与上限的差距
就是蒸馏损失。

## 跑评测

```bash
# 规则测试集（含基线对照）
python src/eval/eval_model.py --model Qwen3.5-0.8B-TextPreprocessor \
    --data data/rule/test.jsonl --baseline --out eval/rule_test.json

# 老师测试集
python src/eval/eval_model.py --model Qwen3.5-0.8B-TextPreprocessor \
    --data data/teacher/test_v2.jsonl --baseline --out eval/teacher_test.json

# 逐条预测（用于人工抽查和生成示例）
python src/eval/eval_model.py --model Qwen3.5-0.8B-TextPreprocessor \
    --data data/rule/test.jsonl --save-preds eval/preds
```

## 怎么看结果

1. **先看 `add`**。如果 `add > 0.02`，说明模型在改写内容，这是比"少删一个空格"严重得多的问题。
2. **再看 `exact` 与 `sim` 的差距**。差距大说明错误集中在少数几个字符上
   （常见是标点全半角、空格位置），这类错误对下游影响小。
3. **最后看失败案例**。`--save-preds` 产出的 JSONL 里逐条都有 raw / ref / pred，
   按 `add` 排序看前 20 条，比看总分有用。

## 复现性

- 数据生成有固定随机种子（`--seed 20261114`），同一命令产出同一份数据。
- 训练固定 `seed=2026`。
- 老师数据生成时 temperature 在 0.5-0.9 之间随机，这部分**不可完全复现**，
  但标签由 oracle 决定，老师的答案只做校验，所以对最终指标影响有限。
