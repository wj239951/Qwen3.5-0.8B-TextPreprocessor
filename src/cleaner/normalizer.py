"""文本清洗规范化引擎（规则 oracle）。

本模块是 docs/TASK_SPEC.md 的可执行实现，同时承担三个角色：

1. 训练数据的**标签生成器**：注入噪声后由本模块产出标准答案，标签精确、可复现；
2. 模型的**规则基线**：与蒸馏出来的 0.8B 模型对比，用来证明学生学到了规则之外的东西；
3. 开源工具：可以脱离模型单独使用，`pip install` 后直接调用 `clean()`。

处理顺序严格遵循规范 §3，顺序不可交换。
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

__all__ = ["clean", "CleanResult", "clean_report", "repair_mojibake", "is_cjk", "normalize_width"]


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

#: 零宽字符与格式控制符：一定会删
_INVISIBLE_ALWAYS = frozenset(
    "\u200b\u200c\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
    "\u2060\u2061\u2062\u2063\u2064\u2066\u2067\u2068\u2069"
    "\ufeff\u00ad\u180e\u034f\u115f\u1160\uffa0\u061c"
)

#: 行内空白类字符：统一成半角空格
_SPACE_LIKE = {
    "\u00a0": " ",  # NO-BREAK SPACE
    "\u2007": " ",  # FIGURE SPACE
    "\u2009": " ",  # THIN SPACE
    "\u200a": " ",  # HAIR SPACE
    "\u202f": " ",  # NARROW NO-BREAK SPACE
    "\u205f": " ",  # MEDIUM MATHEMATICAL SPACE
    "\u3000": " ",  # IDEOGRAPHIC SPACE（全角空格）
    "\t": " ",
}

#: 全角字符 → 半角，只覆盖字母数字和 URL/数学里常见的符号。
#: 中文标点（，。！？：；（）【】等）不在此列，必须保留全角。
_FULLWIDTH_MAP = {
    **{chr(0xFF10 + i): chr(0x30 + i) for i in range(10)},
    **{chr(0xFF21 + i): chr(0x41 + i) for i in range(26)},
    **{chr(0xFF41 + i): chr(0x61 + i) for i in range(26)},
    **{
        "\uff20": "@", "\uff03": "#", "\uff04": "$", "\uff05": "%", "\uff06": "&",
        "\uff0a": "*", "\uff0b": "+", "\uff0d": "-", "\uff0f": "/", "\uff1c": "<",
        "\uff1e": ">", "\uff1d": "=", "\uff3b": "[", "\uff3d": "]", "\uff3e": "^",
        "\uff3f": "_", "\uff5c": "|", "\uff5e": "~", "\uff3c": "\\",
    },
}

#: 中文语境下的半角标点 → 全角
_HALF_TO_CJK_PUNCT = {",": "，", ";": "；", ":": "：", "!": "！", "?": "？", "(": "（", ")": "）"}

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_CJK_CHAR_RUN = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff])\1{2,}")
_LATIN_CHAR_RUN = re.compile(r"([A-Za-z])\1{2,}")

#: 受保护片段：URL / 邮箱 / 行内代码 / 代码围栏 / 图片与链接地址
_PROTECT_PATTERNS = [
    re.compile(r"```.*?```", re.S),
    re.compile(r"`[^`\n]{1,200}`"),
    re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"'，。；）】]+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"!?\[[^\]\n]{0,80}\]\((?:https?://|/)[^)\s]{1,300}\)"),
]

_HTML_TAG = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]{0,9})(?:\s[^<>]{0,300})?/?>")
#: 块级标签 = 段落边界，转成空行
_HTML_PARA_BREAK = re.compile(
    r"</(?:p|div|h[1-6]|section|article|blockquote|pre|table|ul|ol|dl|figure)>", re.I
)
#: 行级标签 = 换行，转成单个换行
_HTML_LINE_BREAK = re.compile(r"<(?:br|hr)\s*/?>|</(?:li|tr|td|th|dt|dd|caption)>", re.I)

_TERMINAL_PUNCT = "。！？!?：:；;，,、…）)]}】」』”\"'"
_LIST_PREFIX = re.compile(r"^\s*(?:[-*+•·]|\d{1,2}[.)、]|#|>|\|)")

#: 乱码里高频出现的“拉丁补充/扩展”字符
_MOJIBAKE_HINTS = frozenset("ÃÂâãäåæçèéêëìíîïðñòóôõöùúûüýþÿÅÆØÐÞŠŽŸ")

#: 常用汉字表（按语料频率取前若干），用于区分“真中文”和“GB18030 假阳性”
_COMMON_CJK = frozenset(
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会"
    "家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从"
    "无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两"
    "次使身者被高已亲其进此话常与活正感见明问力理尔点文几定本公特做外孩相西果走将月十实"
    "向声车全信重三机工物气每并别真打太新比才便夫再书部水像眼等体却加电主界门利海受听表"
    "德少克代员许先口由死安写性马光白或住难望教命花结乐色更拉东神记处让母父应直字场平报"
    "友关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原满性语意力"
)


# --------------------------------------------------------------------------- #
# 结果容器
# --------------------------------------------------------------------------- #

@dataclass
class CleanResult:
    """带诊断信息的清洗结果，方便做数据质检和消融实验。"""

    text: str
    mojibake_fixed: bool = False
    html_detected: bool = False
    invisible_removed: int = 0
    lines_joined: int = 0
    steps: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 基础判定
# --------------------------------------------------------------------------- #

def is_cjk(ch: str) -> bool:
    """是否为中日韩表意字符（用于标点语境判断）。"""
    return bool(ch) and bool(_CJK_RE.match(ch))


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if _CJK_RE.match(ch)) / len(text)


# --------------------------------------------------------------------------- #
# 步骤 1-4：换行、不可见字符、乱码、HTML
# --------------------------------------------------------------------------- #

def normalize_newlines(text: str) -> str:
    """CRLF / CR / 垂直制表 → LF（规范 §3-1）。"""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\v", "\n").replace("\f", "\n")


def strip_invisible(text: str, keep_c1: bool = False) -> tuple[str, int]:
    """删除零宽字符与控制字符，保留 ``\\n``；emoji 组合用的 ZWJ 例外保留。

    ``keep_c1=True`` 时保留 U+0080–U+009F 这一段 C1 控制符：它们是
    "UTF-8 字节被按 Latin-1 解码" 型乱码的原始字节，删掉就再也修不回来了。
    """
    out: list[str] = []
    removed = 0
    for i, ch in enumerate(text):
        if ch in ("\n",):
            out.append(ch)
            continue
        if ch == "\t":
            # 制表符是 Cc，但语义上是分隔符：必须转成空格，直接删会把词粘在一起
            out.append(" ")
            continue
        if ch == "\u200d":  # ZWJ：👨‍👩‍👧 这类组合 emoji 依赖它
            prev_ch = text[i - 1] if i > 0 else ""
            next_ch = text[i + 1] if i + 1 < len(text) else ""
            if _is_emoji(prev_ch) or _is_emoji(next_ch):
                out.append(ch)
            else:
                removed += 1
            continue
        if ch in _INVISIBLE_ALWAYS:
            removed += 1
            continue
        if keep_c1 and 0x80 <= ord(ch) <= 0x9F:
            out.append(ch)
            continue
        if unicodedata.category(ch) in ("Cc", "Cf", "Cs") and ch != "\n":
            removed += 1
            continue
        out.append(ch)
    return "".join(out), removed


def strip_c1(text: str) -> str:
    """清掉残留的 C1 控制符（乱码修复之后调用，此时它们已无用途）。"""
    return "".join(ch for ch in text if not (0x80 <= ord(ch) <= 0x9F))


#: 不可能是「单字节编码字节值」的不可见字符：可以在修乱码之前直接删掉。
#: U+00AD、C1 控制符都 <= 0xFF，它们可能就是乱码的原始字节，必须留到修复之后。
_NONBYTE_INVISIBLE = frozenset(
    "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
    "\u2060\u2061\u2062\u2063\u2064\u2066\u2067\u2068\u2069"
    "\ufeff\u180e\u034f\u115f\u1160\uffa0\u061c"
)


def strip_nonbyte_invisible(text: str) -> str:
    """修复乱码前先摘掉不可能是字节值的零宽字符。

    否则 `latin-1` 编码会整体失败，整段乱码就修不动了（实测造成 4% 的脏标签）。
    ZWJ 例外：组合 emoji 靠它，且它出现时附近一定是 emoji 而不是乱码。
    """
    out = []
    for i, ch in enumerate(text):
        if ch not in _NONBYTE_INVISIBLE:
            out.append(ch)
            continue
        if ch == "\u200d":
            prev_ch = text[i - 1] if i > 0 else ""
            next_ch = text[i + 1] if i + 1 < len(text) else ""
            if _is_emoji(prev_ch) or _is_emoji(next_ch):
                out.append(ch)
    return "".join(out)


def _is_emoji(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    return (
        0x1F000 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
        or 0x2B00 <= cp <= 0x2BFF
        or 0xFE00 <= cp <= 0xFE0F
    )


def repair_mojibake(text: str) -> tuple[str, bool]:
    """修复 UTF-8 被误按 CP1252 / Latin-1 解码产生的乱码。

    只在高置信时动手：候选结果必须显著提高 CJK 比例且不引入替换符，
    否则原样返回。无法修复的 ``锟斤拷`` 残渣直接删除。
    """
    cleaned = text.replace("锟斤拷", "")
    if not any(ch in _MOJIBAKE_HINTS for ch in cleaned):
        return cleaned, cleaned != text
    fixed = _repair_segment(cleaned)
    if fixed is not None:
        return fixed, True

    # 整段不可编码时，按行重试（常见于"正文正常 + 个别行乱码"的混合文本）
    lines = cleaned.split("\n")
    changed = False
    new_lines = []
    for line in lines:
        candidate = _repair_segment(line) if any(c in _MOJIBAKE_HINTS for c in line) else None
        if candidate is not None:
            new_lines.append(candidate)
            changed = True
        else:
            new_lines.append(line)
    if changed:
        return "\n".join(new_lines), True

    # 行内混排（中文 + 乱码片段）时，按"疑似乱码片段"逐段重试
    repaired = _repair_runs(cleaned)
    if repaired != cleaned:
        return repaired, True
    return cleaned, cleaned != text


#: 连续 4 个以上的非 CJK 非空白字符，可以整段尝试编码往返
_RUN_RE = re.compile(r"[^ \n\t\r\f\v\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]{4,}")


def _repair_runs(text: str) -> str:
    def _one(match: re.Match) -> str:
        run = match.group(0)
        if not any(ch in _MOJIBAKE_HINTS for ch in run):
            return run
        fixed = _repair_segment(run)
        return fixed if fixed is not None else run

    return _RUN_RE.sub(_one, text)


def _repair_segment(segment: str) -> str | None:
    """尝试若干编码往返，返回得分最优且明显改善的候选。

    UTF-8 被误按单字节编码解码是最常见的乱码来源，因此 utf-8 候选带优先级加成；
    GB18030 候选只作为兜底，并要求质量提升更多，避免把正常文本“修”成生僻字。
    """
    if not segment.strip():
        return None
    base_score = _text_quality(segment)
    best: tuple[float, str] | None = None
    # 只尝试 UTF-8：GBK/GB18030 型"二次损坏"解码成功率低且极易把正常文本
    # 解成生僻字乱码（实测污染了 4% 的标签），宁可修不了也不要修错。
    for enc_from in ("cp1252", "latin-1"):
        try:
            candidate = segment.encode(enc_from, errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
            continue
        if "\ufffd" in candidate:
            continue
        score = _text_quality(candidate)
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None:
        return None
    return best[1] if best[0] > base_score + 0.15 else None


def _text_quality(text: str) -> float:
    """粗粒度“像正常文本”打分：常用汉字比例高、乱码提示字符少、无替换符。"""
    if not text:
        return 0.0
    n = len(text)
    cjk = _cjk_ratio(text)
    common = sum(1 for ch in text if ch in _COMMON_CJK) / n
    hint = sum(1 for ch in text if ch in _MOJIBAKE_HINTS) / n
    replacement = 1.0 if "\ufffd" in text else 0.0
    return cjk * 0.6 + common * 0.6 + (0.1 if text.strip() else 0.0) - hint * 2.0 - replacement


def html_to_text(text: str) -> tuple[str, bool]:
    """HTML → 纯文本：实体解码、块级标签转换行、其余标签删除。

    只有检测到 ``>=2`` 个成对标签结构时才认为输入是 HTML，
    避免把 ``a < b and c > d`` 这类数学表达式误删（规范 §6）。
    """
    tags = _HTML_TAG.findall(text)
    # 只要有 <br>/<p> 这类块级结构，或出现 >=2 个标签，就按 HTML 处理
    looks_like_html = bool(_HTML_LINE_BREAK.search(text)) or bool(_HTML_PARA_BREAK.search(text)) or len(tags) >= 2
    if not looks_like_html:
        if "&" in text and re.search(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d{2,5});", text):
            return html.unescape(text), False
        return text, False
    text = html.unescape(text)
    text = _HTML_PARA_BREAK.sub("\n\n", text)
    text = _HTML_LINE_BREAK.sub("\n", text)
    text = _HTML_TAG.sub("", text)
    return text, True


# --------------------------------------------------------------------------- #
# 步骤 5-7：宽度、标点、重复
# --------------------------------------------------------------------------- #

def normalize_space_chars(text: str) -> str:
    """各类不间断空格 / 窄空格 / 全角空格 / 制表符 → 半角空格（规范 §3-5）。

    按 Unicode 类别兜底：所有 ``Zs``（分隔符空格）都归一，避免 U+2002、U+2009
    这类冷门空格漏网。
    """
    for src, dst in _SPACE_LIKE.items():
        text = text.replace(src, dst)
    return "".join(
        " " if (ch != " " and unicodedata.category(ch) == "Zs") else ch for ch in text
    )


def normalize_width(text: str) -> str:
    """全角字母数字 → 半角，全角/特殊空格 → 半角空格（规范 §3-5）。"""
    text = normalize_space_chars(text)
    return "".join(_FULLWIDTH_MAP.get(ch, ch) for ch in text)


def normalize_punct(text: str) -> str:
    """中文语境标点归一 + 连续标点压缩（规范 §3-6、§2.2）。"""
    # 半角标点被 CJK 包围时转全角
    chars = list(text)
    for i, ch in enumerate(chars):
        target = _HALF_TO_CJK_PUNCT.get(ch)
        if target is None:
            continue
        prev_ch = chars[i - 1] if i > 0 else ""
        next_ch = chars[i + 1] if i + 1 < len(chars) else ""
        if not is_cjk(prev_ch):
            continue
        if ch in "!?" or ch in "()" or ch in ",;:":
            if is_cjk(next_ch) or next_ch in ("", "\n", " "):
                chars[i] = target
    text = "".join(chars)

    # 连续标点压缩
    text = re.sub(r"([!?！？])\1+", r"\1", text)
    text = re.sub(r"。{3,}", "……", text)
    text = re.sub(r"。{2}", "。", text)
    text = re.sub(r"…{2,}", "……", text)
    text = re.sub(r"([，,、；;：:])\1+", r"\1", text)
    text = re.sub(r"\.{3,}", "……", text) if _cjk_ratio(text) > 0.3 else text
    return text


def collapse_repeats(text: str) -> str:
    """压缩无意义重复字符（规范 §3-7、§2.3）。数字不处理。"""
    text = _CJK_CHAR_RUN.sub(r"\1", text)
    text = _LATIN_CHAR_RUN.sub(r"\1\1", text)
    return text


# --------------------------------------------------------------------------- #
# 步骤 8-11：行、硬换行、空白、首尾
# --------------------------------------------------------------------------- #

def _strip_lines(text: str) -> str:
    return "\n".join(line.strip() for line in text.split("\n"))


def _should_join(prev_line: str, next_line: str) -> bool:
    """规范 §4 的四个条件，全部满足才合并。"""
    if not prev_line or not next_line:
        return False
    if len(prev_line) < 24:
        return False
    if prev_line[-1] in _TERMINAL_PUNCT:
        return False
    if _LIST_PREFIX.match(next_line):
        return False
    if not re.search(r"[0-9A-Za-z\u4e00-\u9fff]", next_line):
        return False
    return True


def _merge_lines(prev_line: str, next_line: str) -> str:
    left_cjk = is_cjk(prev_line[-1])
    right_cjk = is_cjk(next_line[0])
    if left_cjk or right_cjk:
        return prev_line + next_line
    return prev_line + " " + next_line


def join_hard_wraps(text: str) -> tuple[str, int]:
    """合并段落内被排版折断的行（规范 §3-9、§4）。

    两级判定：
    1. **段落级**——同段落里多行长度接近、且除末行外都不以终止标点收尾，
       判定为"排版折断"，整段合并。这是 PDF / 终端复制最常见的形态。
    2. **逐对**——段落级不成立时退回保守的单行判定（长度 ≥24、下一行不像新句子）。
    """
    lines = text.split("\n")
    out: list[str] = []
    joined = 0
    i = 0
    while i < len(lines):
        block_start = i
        while i < len(lines) and lines[i].strip():
            i += 1
        block = lines[block_start:i]
        merged, n = _merge_block(block)
        out.extend(merged)
        joined += n
        if i < len(lines):
            out.append(lines[i])
            i += 1
    return "\n".join(out), joined


def _merge_block(block: list[str]) -> tuple[list[str], int]:
    """段落级硬换行合并，返回 (新行列表, 合并次数)。

    固定宽度排版（PDF / 终端 / 打印）留下的签名是：**除最后一行外，行长高度一致**，
    且句子都没说完。只有同时满足这两点才整段合并——这是能看到的最强证据。
    两行的情况证据不足，退回保守的逐对判定。
    """
    if len(block) < 2:
        return block, 0
    if any(_LIST_PREFIX.match(line) for line in block):
        return _merge_pairwise(block)

    lengths = [len(line) for line in block]
    if len(block) == 2:
        return _merge_pairwise(block)

    head = lengths[:-1]
    median = sorted(head)[len(head) // 2]
    if median < 16:
        return _merge_pairwise(block)
    if (max(head) - min(head)) / median > 0.25:
        return _merge_pairwise(block)
    if any(block[idx][-1] in _TERMINAL_PUNCT for idx in range(len(block) - 1)):
        return _merge_pairwise(block)

    merged_text = block[0]
    for line in block[1:]:
        merged_text = _merge_lines(merged_text, line)
    return [merged_text], len(block) - 1


def _merge_pairwise(block: list[str]) -> tuple[list[str], int]:
    """保守的逐行合并：只在条件明确时合并。"""
    out: list[str] = []
    joined = 0
    for line in block:
        if out and out[-1] and _should_join(out[-1], line):
            out[-1] = _merge_lines(out[-1], line)
            joined += 1
        else:
            out.append(line)
    return out, joined


def normalize_space(text: str) -> str:
    """压缩连续空格与连续空行（规范 §3-10）。"""
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[ ]*\n[ ]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 保护片段
# --------------------------------------------------------------------------- #

class _SpanGuard:
    """用私有区占位符保护 URL / 邮箱 / 代码，避免被宽度、标点规则改写。"""

    _OPEN, _CLOSE = "\ue000", "\ue001"

    def __init__(self) -> None:
        self.spans: list[str] = []

    def protect(self, text: str) -> str:
        for pattern in _PROTECT_PATTERNS:
            text = pattern.sub(self._replace, text)
        return text

    def _replace(self, match: re.Match) -> str:
        self.spans.append(match.group(0))
        return f"{self._OPEN}{len(self.spans) - 1}{self._CLOSE}"

    def restore(self, text: str) -> str:
        def _back(match: re.Match) -> str:
            idx = int(match.group(1))
            return self.spans[idx] if 0 <= idx < len(self.spans) else match.group(0)

        return re.sub(rf"{self._OPEN}(\d+){self._CLOSE}", _back, text)


# --------------------------------------------------------------------------- #
# 对外主函数
# --------------------------------------------------------------------------- #

def clean(text: str) -> str:
    """清洗规范化主入口。空输入返回空串（规范 §2.5）。"""
    return clean_report(text).text


def clean_report(text: str) -> CleanResult:
    """清洗并返回诊断信息。"""
    result = CleanResult(text="")
    if text is None:
        return result
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return result

    # 1) 换行统一
    text = normalize_newlines(text)

    # 2) 先摘掉不可能是字节值的零宽字符，再修乱码
    #    （乱码的原始字节里包含 C1 控制符与 U+00AD，但它们必须活到修复之后）
    text = strip_nonbyte_invisible(text)
    text, fixed = repair_mojibake(text)
    result.mojibake_fixed = fixed

    # 3) 不可见字符 + 残留 C1
    text, removed = strip_invisible(text)
    result.invisible_removed = removed
    text = strip_c1(text)

    # 4) HTML
    text, is_html = html_to_text(text)
    result.html_detected = is_html

    # 4.5) 空白类字符先统一（含 NBSP / 全角空格），避免它们藏进受保护片段里
    text = normalize_space_chars(text)

    # 保护 URL / 邮箱 / 代码
    guard = _SpanGuard()
    text = guard.protect(text)

    # 5) 宽度  6) 标点  7) 重复
    text = normalize_width(text)
    text = normalize_punct(text)
    text = collapse_repeats(text)

    # 8) 行级清理  9) 硬换行合并
    text = _strip_lines(text)
    text, joined = join_hard_wraps(text)
    result.lines_joined = joined

    # 10) 空白  11) 首尾
    text = normalize_space(text)

    # 还原保护片段
    text = guard.restore(text)
    result.text = text.strip()
    if not result.text:
        result.text = ""
    return result
