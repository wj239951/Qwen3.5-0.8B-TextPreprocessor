"""清洗引擎的契约测试：每条用例都对应 docs/TASK_SPEC.md 的一节。

直接运行（无需第三方依赖）：
    python tests/test_normalizer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cleaner.normalizer import clean, clean_report  # noqa: E402

CASES: list[tuple[str, str, str]] = [
    # (说明, 输入, 期望输出)
    ("零宽字符删除", "你\u200b好\u200c世\u200d界", "你好世界"),
    ("BOM 与软连字符", "\ufeffhello\u00adworld", "helloworld"),
    ("全角字母数字", "Ｈｅｌｌｏ　１２３", "Hello 123"),
    ("全角空格归一", "中文\u3000换行", "中文 换行"),
    # 规范 §2.3：只压缩 >=3 次重复；"谢谢/妈妈/刚刚"这类两字叠词是合法词，必须保留
    ("CJK 重复压缩", "我我我明天去去北京", "我明天去去北京"),
    ("拉丁重复压缩", "sooooo good", "soo good"),
    ("数字不压缩", "订单号 1000000", "订单号 1000000"),
    ("CJK 间半角标点转全角", "你好,世界!", "你好，世界！"),
    ("拉丁语境标点不动", "version 1.2.3, ok", "version 1.2.3, ok"),
    ("感叹号压缩", "太好了！！！", "太好了！"),
    ("连续句号转省略号", "我想想。。。", "我想想……"),
    ("多余空白压缩", "  今天   天气   不错  ", "今天 天气 不错"),
    ("连续空行压缩", "第一段\n\n\n\n\n第二段", "第一段\n\n第二段"),
    ("行尾行首空白", "  第一行   \n   第二行  ", "第一行\n第二行"),
    (
        "硬换行合并",
        "这是一段被排版软件强行折断的长句子，它在第二十三\n个字符处断开了，需要合并回一行。",
        "这是一段被排版软件强行折断的长句子，它在第二十三个字符处断开了，需要合并回一行。",
    ),
    (
        "PDF 三行折断合并",
        "这是一段从PDF复制的文字，每行都被折断\n所以看起来是这样的，读起来很不连贯\n需要合并成一段。",
        "这是一段从PDF复制的文字，每行都被折断所以看起来是这样的，读起来很不连贯需要合并成一段。",
    ),
    ("两行聊天不合并", "我刚看到你发的那个方案，感觉还可以\n不过预算这块还需要再确认一下",
     "我刚看到你发的那个方案，感觉还可以\n不过预算这块还需要再确认一下"),
    ("诗句不合并", "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡",
     "床前明月光\n疑是地上霜\n举头望明月\n低头思故乡"),
    ("列表行不合并", "下面是要点：\n- 第一点\n- 第二点", "下面是要点：\n- 第一点\n- 第二点"),
    ("短行不合并", "好的\n我知道了", "好的\n我知道了"),
    (
        "URL 保护",
        "请看 https://example.com/a?b=1&c=2 这里",
        "请看 https://example.com/a?b=1&c=2 这里",
    ),
    (
        "邮箱保护",
        "联系 me.test+tag@example.co.uk 谢谢",
        "联系 me.test+tag@example.co.uk 谢谢",
    ),
    ("行内代码保护", "运行 `pip install foo-bar` 即可", "运行 `pip install foo-bar` 即可"),
    ("HTML 标签", "<p>你好</p><p>世界</p>", "你好\n\n世界"),
    ("HTML div 分段", "<div>甲</div><div>乙</div>", "甲\n\n乙"),
    ("HTML 列表项只换行", "<ul><li>一</li><li>二</li></ul>", "一\n二"),
    ("HTML 实体", "a &amp; b &lt; c", "a & b < c"),
    ("HTML br 转换行", "第一行<br/>第二行", "第一行\n第二行"),
    ("非 HTML 的尖括号保留", "若 a < b 且 c > d 则成立", "若 a < b 且 c > d 则成立"),
    ("乱码修复", "æ˜Žå¤©åŽ»åŒ—äº¬", "明天去北京"),
    ("空输入", "   \n\t\u200b ", ""),
    ("纯不可见字符", "\u200b\ufeff\u200c", ""),
    ("emoji 组合保留", "👨\u200d👩\u200d👧 一家人", "👨\u200d👩\u200d👧 一家人"),
    ("换行符统一", "第一行\r\n第二行\r第三行", "第一行\n第二行\n第三行"),
    ("句末标点后不合并", "句子结束了。\n下一句开始", "句子结束了。\n下一句开始"),
]


def main() -> int:
    failed = 0
    for name, src, expected in CASES:
        actual = clean(src)
        if actual != expected:
            failed += 1
            print(f"FAIL  {name}")
            print(f"   in : {src!r}")
            print(f"   out: {actual!r}")
            print(f"   exp: {expected!r}")
        else:
            print(f"PASS  {name}")

    report = clean_report("<p>你好</p>\n\n\n\n  世界\u200b  ")
    print(f"\nreport: html={report.html_detected} invisible={report.invisible_removed} -> {report.text!r}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
