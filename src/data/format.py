"""训练/推理共用的样本格式与提示词。

系统提示词是模型契约的一部分：训练、评测、推理、模型卡必须完全一致，
否则学生看到的分布会偏移。
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "你是输入文本预处理器。对用户给出的原始文本做清洗规范化："
    "删除不可见字符与噪声、修复乱码、去掉 HTML 标签、统一全角半角与标点、"
    "压缩无意义重复、整理空白与断行。只改格式不改内容，"
    "直接输出清洗后的文本，不要解释。"
)

#: 训练时使用的对话模板（assistant 段参与 loss）
def to_messages(raw: str, clean: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw},
        {"role": "assistant", "content": clean},
    ]


#: 推理时的对话模板（不带 assistant 段）
def to_prompt_messages(raw: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw},
    ]
