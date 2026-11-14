"""噪声注入器：把干净文本「弄脏」，用于合成训练数据。

每个注入器对应 docs/TASK_SPEC.md 中的一个清洗步骤，是它的逆操作：

| 注入器                | 对应的清洗步骤        |
|-----------------------|-----------------------|
| ``zero_width``        | 删除不可见字符        |
| ``control_chars``     | 删除控制字符          |
| ``fullwidth``         | 宽度归一              |
| ``halfwidth_punct``   | 标点归一              |
| ``char_repeat``       | 重复字符压缩          |
| ``punct_repeat``      | 连续标点压缩          |
| ``messy_space``       | 空白归一              |
| ``blank_lines``       | 空行归一 + 换行统一   |
| ``html_wrap``         | HTML 处理             |
| ``mojibake``          | 乱码修复              |
| ``hard_wrap``         | 硬换行合并            |

用法：
    noisy, kinds = corrupt(clean_text, rng, level="medium")
"""

from __future__ import annotations

import random
import re

__all__ = ["corrupt", "INJECTORS"]

_INVISIBLE_POOL = ["\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad", "\u202e"]
#: 只放不会改变行结构的控制符（\x0b / \x0c 会被当成换行，不能用来模拟噪声）
_CONTROL_POOL = ["\x00", "\x01", "\x07", "\x08", "\x1f"]
_FULLWIDTH_OFFSET = 0xFEE0
_HALF_TO_CJK = {",": "，", ";": "；", ":": "：", "!": "！", "?": "？", "(": "（", ")": "）"}
_CJK = re.compile(r"[\u4e00-\u9fff]")
#: 需要整体跳过的 URL / 邮箱（必须匹配完整路径，只匹配前缀会把主机名改坏）
_URLISH = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"'，。；）】]+|[\w.+-]+@[\w-]+\.[\w.-]+")
#: 这些字符后面是句子/分句边界，紧跟换行会被规范的“硬换行合并”拒绝
_NO_CUT_AFTER = set("。！？!?：:；;，,、…）)]}\"'")


# --------------------------------------------------------------------------- #
# 单个注入器
# --------------------------------------------------------------------------- #

def zero_width(text: str, rng: random.Random) -> str:
    """随机插入零宽字符。"""
    chars = list(text)
    for _ in range(rng.randint(1, 4)):
        pos = rng.randrange(len(chars) + 1)
        chars.insert(pos, rng.choice(_INVISIBLE_POOL))
    return "".join(chars)


def control_chars(text: str, rng: random.Random) -> str:
    """插入控制字符。"""
    chars = list(text)
    for _ in range(rng.randint(1, 3)):
        pos = rng.randrange(len(chars) + 1)
        chars.insert(pos, rng.choice(_CONTROL_POOL))
    return "".join(chars)


def _protected_spans(text: str) -> list[tuple[int, int]]:
    """代码围栏 / 行内代码 / URL / 邮箱的位置，这些片段不能被噪声破坏。"""
    spans = [m.span() for m in re.finditer(r"```.*?```", text, re.S)]
    spans += [m.span() for m in _URLISH.finditer(text)]
    spans += [m.span() for m in re.finditer(r"`[^`\n]{1,200}`", text)]
    return spans


def _in_spans(i: int, spans: list[tuple[int, int]]) -> bool:
    return any(lo <= i < hi for lo, hi in spans)


def fullwidth(text: str, rng: random.Random) -> str:
    """把一部分 ASCII 字母数字转成全角，部分空格转全角空格。

    只动字母数字：标点一旦被全角化，就会与中文标点混淆，规范里无法无损还原。
    URL / 邮箱 / 代码片段整体跳过，避免破坏受保护内容。
    """
    spans = _protected_spans(text)
    chars: list[str] = []
    for i, ch in enumerate(text):
        if _in_spans(i, spans):
            chars.append(ch)
        elif ch.isascii() and ch.isalnum() and rng.random() < 0.55:
            chars.append(chr(ord(ch) + _FULLWIDTH_OFFSET))
        elif ch == " " and rng.random() < 0.25:
            chars.append("\u3000")
        else:
            chars.append(ch)
    return "".join(chars)


def halfwidth_punct(text: str, rng: random.Random) -> str:
    """把中文语境里的全角标点降级成半角，模拟输入法混用。"""
    back = {v: k for k, v in _HALF_TO_CJK.items()}
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in back and rng.random() < 0.6 and i > 0 and _CJK.match(chars[i - 1]):
            chars[i] = back[ch]
    return "".join(chars)


def char_repeat(text: str, rng: random.Random) -> str:
    """把随机几个汉字重复 2-4 次，模拟手抖或输入法重复。

    只挑**左右都不与自己相同**的字，否则会把"谢谢"这类合法叠词改成 3 连击，
    导致规范把它压缩成单字，标签失真。
    """
    positions = [
        i
        for i, ch in enumerate(text)
        if _CJK.match(ch)
        and (i == 0 or text[i - 1] != ch)
        and (i + 1 >= len(text) or text[i + 1] != ch)
    ]
    if not positions:
        return text
    for pos in sorted(rng.sample(positions, min(len(positions), rng.randint(1, 3))), reverse=True):
        if rng.random() < 0.75:
            # 至少补 2 个：连击长度必须 >=3 才会被规范压缩成 1，
            # 只补 1 个会造出规范明确保留的"两字叠词"
            text = text[: pos + 1] + text[pos] * rng.randint(2, 3) + text[pos + 1 :]
    return text


def punct_repeat(text: str, rng: random.Random) -> str:
    """标点重复：！→！！、。→。。（两个即止，三个以上会被规范改写成省略号）。"""
    text = re.sub(r"([！!])", lambda m: m.group(1) * rng.randint(2, 3), text)
    text = re.sub(r"(。)(?=[\s\n]|$)", "。。", text)
    return text


def messy_space(text: str, rng: random.Random) -> str:
    """空格污染：首尾空格、连续空格、制表符（受保护片段内部不动）。"""
    spans = _protected_spans(text)
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch == " " and not _in_spans(i, spans) and rng.random() < 0.3:
            out.append(" " * rng.randint(2, 3))
        elif ch == " " and not _in_spans(i, spans) and rng.random() < 0.15:
            out.append("\t")
        else:
            out.append(ch)
    text = "".join(out)
    return " " * rng.randint(1, 3) + text + " " * rng.randint(1, 3)


def blank_lines(text: str, rng: random.Random) -> str:
    """空行膨胀 + 换行符风格混乱。"""
    text = re.sub(r"\n\n", "\n" * rng.randint(3, 6), text)
    if rng.random() < 0.5:
        text = text.replace("\n", "\r\n")
    elif rng.random() < 0.3:
        text = text.replace("\n", "\r")
    return text


def html_wrap(text: str, rng: random.Random) -> str:
    """包一层 HTML：段落标签、换行标签、实体。"""
    paragraphs = text.split("\n\n") if "\n\n" in text else [text]
    parts = []
    for para in paragraphs:
        body = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if rng.random() < 0.4:
            body = body.replace(" ", "&nbsp;", rng.randint(1, 3))
        if rng.random() < 0.5 and "\n" in body:
            body = body.replace("\n", "<br/>")
        tag = rng.choice(["p", "div"])
        parts.append(f"<{tag}>{body}</{tag}>")
    return "\n".join(parts)


def mojibake(text: str, rng: random.Random) -> str:
    """UTF-8 字节被按 Latin-1 解码，产生 ``æ˜Žå¤©`` 型乱码。"""
    if text.isascii():
        return text
    # 只把 ASCII 空白当成"分隔符"跳过。全角空格 U+3000、NBSP 这类也必须编码进乱码，
    # 否则它们会留在字符串里，让整段 UTF-8 字节流断开、永远修不回来。
    out = []
    for chunk in re.split(r"([ \t\r\n]+)", text):
        if chunk.strip() and not chunk.isascii():
            out.append(chunk.encode("utf-8").decode("latin-1"))
        else:
            out.append(chunk)
    return "".join(out)


def hard_wrap(text: str, rng: random.Random) -> str:
    """在段落中间插入换行，模拟 PDF / 终端复制产生的折断。

    只在**空格处**或**汉字之间**下刀：切在英文单词内部会造出 "G B" 这种
    无法判定该不该补空格的样本，标签会失真。
    """
    width = rng.choice([28, 32, 36])
    lines = []
    for para in text.split("\n"):
        if len(para) <= width + 8 or _URLISH.search(para) or "`" in para:
            lines.append(para)
            continue
        buf = para
        while len(buf) > width + 8:
            cuts = []
            for i in range(width, min(width + 10, len(buf) - 4)):
                if buf[i - 1] in _NO_CUT_AFTER:
                    continue
                # 要么切在“拉丁词与拉丁词之间”的空格上（合并时补回空格），
                # 要么切在“汉字与汉字之间”（合并时不补空格）。其他位置不切，
                # 否则合并规则无法判断该不该补空格，标签会失真。
                if buf[i] == " " and buf[i - 1].isascii() and buf[i + 1 : i + 2].isascii():
                    cuts.append((i, True))
                elif _CJK.match(buf[i]) and _CJK.match(buf[i - 1]):
                    cuts.append((i, False))
            if not cuts:
                break
            cut, eat_space = cuts[0]
            lines.append(buf[:cut])
            buf = buf[cut + 1 :] if eat_space else buf[cut:]
        lines.append(buf)
    return "\n".join(lines)


INJECTORS = {
    "zero_width": zero_width,
    "control_chars": control_chars,
    "fullwidth": fullwidth,
    "halfwidth_punct": halfwidth_punct,
    "char_repeat": char_repeat,
    "punct_repeat": punct_repeat,
    "messy_space": messy_space,
    "blank_lines": blank_lines,
    "html_wrap": html_wrap,
    "mojibake": mojibake,
    "hard_wrap": hard_wrap,
}

#: 训练时可用的注入器。乱码被排除：字节级还原交给 repair_mojibake()，
#: 小模型学不会，硬教只会让它学会编造。
CORRUPT_INJECTORS = {k: v for k, v in INJECTORS.items() if k != "mojibake"}

#: 难度档位 → (最少注入器个数, 最多注入器个数)
_LEVELS = {"light": (1, 1), "medium": (2, 3), "heavy": (3, 5)}


def corrupt(text: str, rng: random.Random, level: str = "medium") -> tuple[str, list[str]]:
    """对文本施加随机噪声，返回 ``(噪声文本, 注入器名列表)``。

    ``level`` 取 ``light`` / ``medium`` / ``heavy``，控制叠加的噪声种类数量。

    关键约束：``mojibake`` 必须放在最后。乱码是"字符被换成完全不同的字符"，
    任何后续注入（全角化、重复字符、换行折断）都会毁掉原始字节序列，
    导致这段乱码永远修不回来——那会污染标签。
    """
    lo, hi = _LEVELS.get(level, _LEVELS["medium"])
    n = rng.randint(lo, hi)
    names = rng.sample(list(CORRUPT_INJECTORS), min(n, len(CORRUPT_INJECTORS)))
    noisy = text
    applied: list[str] = []
    for name in names:
        try:
            candidate = CORRUPT_INJECTORS[name](noisy, rng)
        except Exception:  # 注入失败不应中断数据生成
            continue
        if candidate != noisy or name in ("zero_width", "control_chars"):
            noisy = candidate
            applied.append(name)
    if not applied:  # 兜底，保证至少有一种噪声
        noisy = messy_space(noisy, rng)
        applied = ["messy_space"]
    return noisy, applied
