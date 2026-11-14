"""用老师模型（DeepSeek-V4.1-Flash，即 ``deepseek-flash``）生成/标注清洗数据。

两件事：

1. ``gen`` 模式——老师**造**样本：给定场景与主题，写出"真实会遇到的脏输入"
   和它清洗后的结果。这样得到的噪声分布比随机注入更自然（真实的 HTML 残留、
   全角半角混排、无意义重复、PDF 折断）。
2. ``label`` 模式——老师**标**样本：把已经有脏输入交给老师清洗，拿到目标输出。

无论哪种模式，老师给的 ``clean`` 都会用 ``cleaner.normalizer``（规则 oracle）
交叉校验：两者相似度足够高才采用，否则记入 rejects。这样既保留了老师的
自然性，又保证标签与 docs/TASK_SPEC.md 不冲突。

用法：
    python src/data/teacher_pairs.py --mode gen --n 120 --workers 8 --out data/teacher/gen.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cleaner.normalizer import clean  # noqa: E402

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

SCENARIOS = [
    "从网页复制的中文文章，带 HTML 标签和实体",
    "从 PDF 复制的中文段落，有硬换行和页眉残留",
    "微信聊天记录转发，有大量空格和重复标点",
    "程序员粘贴的代码与报错日志混排",
    "Excel 单元格复制出的表格行，用制表符分隔",
    "语音转文字结果，有语气词重复和标点缺失",
    "中英混排的技术文档片段，全角半角混用",
    "客服工单，用户输入很随意，有零宽字符",
    "简历中的工作经历段落，含日期和数字",
    "论文摘要中的段落，被排版软件折断",
    "邮件引用内容，带 > 引用符和多层缩进",
    "字幕文件内容，时间轴残留和重复行",
    "商品详情页文案，含 emoji 和促销符号",
    "新闻快讯，从客户端复制带有换行符污染",
    "接口返回的错误信息，夹杂英文和中文",
    "OCR 识别出的说明书段落，标点混乱但字没错",
]

TOPICS = [
    "城市天气与出行建议", "机器学习模型部署", "家庭理财与预算", "儿童疫苗接种",
    "装修预算与工期", "手机选购对比", "咖啡冲泡方法", "项目风险管理",
    "健身计划安排", "租房合同条款", "开源许可证选择", "数据库索引优化",
    "旅游行程规划", "面试准备建议", "慢性病日常护理", "新能源汽车充电",
    "考研复习节奏", "餐厅点单与结账", "宠物饲养注意事项", "摄影构图技巧",
]

SYSTEM_PROMPT = """你是文本清洗数据集的构造专家。你的任务是造出"真实世界里会碰到的脏文本"，
以及它按固定规范清洗后的结果。

清洗规范（只做这些，绝不越界）：
1. 删除零宽字符、控制字符、软连字符；
2. 修复「UTF-8 被按 Latin-1 解码」型乱码（如 æ˜Žå¤© → 明天）；
3. 去 HTML 标签、解码实体（&amp; &nbsp; 等），块级标签转换行；
4. 全角字母数字 → 半角，全角空格/NBSP → 半角空格；
5. 中文语境下半角 , ; : ! ? → 全角；连续 ！/？ 压缩成一个；。。 压缩成 ；
6. 同一汉字连续 3 次以上压缩成 1 个；同一字母连续 3 次以上压缩成 2 个；数字不动；
7. 行首行尾空白删除，连续空格压缩成 1 个，连续空行压缩成 1 个；
8. 段落内被排版折断的行合并回一行。

绝对禁止：改写措辞、纠错错别字、翻译、繁简转换、总结、扩写、脱敏、增删 emoji、
调整段落顺序、输出任何解释。

输出必须是严格 JSON，形如：
{"raw": "脏文本", "clean": "清洗后的文本", "noise_types": ["html", "zero_width"]}
不要输出 JSON 以外的任何字符。"""

FEW_SHOT = """示例 1
场景：从网页复制的中文文章，带 HTML 标签和实体
{"raw": "<p>今天的天气&nbsp;很好，适合出去走走。</p><p>不过下午可能会下雨。</p>", "clean": "今天的天气 很好，适合出去走走。\\n\\n不过下午可能会下雨。", "noise_types": ["html"]}

示例 2
场景：微信聊天记录转发，有大量空格和重复标点
{"raw": "  我我我  明天   去北京 ！！！  ", "clean": "我明天 去北京！", "noise_types": ["zero_width", "repeats", "spaces"]}"""


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #

def _api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")
    return key


def call_teacher(messages: list[dict], *, temperature: float = 0.7, max_tokens: int = 1200,
                 retries: int = 4) -> str:
    """调用老师模型，返回 ``content`` 文本（已关闭思维链）。"""
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "stream": False,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'ignore')}"
            if exc.code in (401, 403):
                raise
            time.sleep(2 ** attempt)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    raise RuntimeError(f"老师调用失败: {last_error}")


# --------------------------------------------------------------------------- #
# 解析与校验
# --------------------------------------------------------------------------- #

_JSON_BLOCK = re.compile(r"\{.*\}", re.S)
_BAD_CLEAN_MARKERS = ("清洗后", "原始文本", "说明：", "解释：", "```")


def parse_pair(content: str) -> dict | None:
    """从模型输出里抠出 JSON，容忍偶发的代码围栏与前后废话。"""
    if not content:
        return None
    match = _JSON_BLOCK.search(content)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "raw" not in obj or "clean" not in obj:
        return None
    raw, cleaned = obj.get("raw"), obj.get("clean")
    if not isinstance(raw, str) or not isinstance(cleaned, str):
        return None
    if not raw.strip() or not cleaned.strip():
        return None
    return {"raw": raw, "clean": cleaned, "noise_types": obj.get("noise_types") or []}


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def validate(pair: dict, *, min_similarity: float = 0.90) -> tuple[dict | None, str]:
    """用规则 oracle 交叉校验老师给的标签。

    返回 (样本, 拒绝原因)。拒绝原因非空时样本为 None。
    """
    raw = pair["raw"]
    teacher_clean = pair["clean"].strip()
    for marker in _BAD_CLEAN_MARKERS:
        if marker in teacher_clean:
            return None, "clean 里混入了解释或围栏"
    oracle_clean = clean(raw)
    if not oracle_clean:
        return None, "规则 oracle 认为清洗后为空（可能整段都是噪声）"
    if not (0.5 <= len(teacher_clean) / max(1, len(oracle_clean)) <= 2.0):
        return None, "clean 与 raw 长度比例异常"
    sim = _similar(oracle_clean, teacher_clean)
    if sim < min_similarity:
        return None, f"与规则 oracle 分歧过大 sim={sim:.2f}"
    # 原始输入必须真的"脏"，否则样本没有信息量
    if clean(raw) == raw:
        return None, "raw 本身已经干净，无需清洗"
    return {
        "raw": raw,
        "clean": oracle_clean,  # 标签以规范为准，老师输出用于校验
        "teacher_clean": teacher_clean,
        "similarity": round(sim, 4),
        "noise_types": pair.get("noise_types", []),
        "src": "teacher",
    }, ""


# --------------------------------------------------------------------------- #
# 生成 / 标注
# --------------------------------------------------------------------------- #

def build_gen_messages(rng: random.Random) -> list[dict]:
    scenario = rng.choice(SCENARIOS)
    topic = rng.choice(TOPICS)
    length = rng.choice(["两三句话", "一个短段落", "两个短段落", "一段带列表的文字"])
    user = (
        f"{FEW_SHOT}\n\n"
        f"现在请按同样的格式造一条新样本。\n"
        f"场景：{scenario}\n主题：{topic}\n长度：{length}\n"
        f"要求：raw 必须带 2-4 种真实噪声（不要只用空格）；clean 严格按规范清洗；"
        f"只输出 JSON。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_label_messages(raw: str) -> list[dict]:
    user = (
        f"请按规范清洗下面这段文本，只输出清洗后的正文，不要解释、不要加引号。\n"
        f"规范：删不可见字符/控制符；修 Latin-1 乱码；去 HTML 标签与实体；"
        f"全角字母数字转半角；中文里的半角标点转全角；连续标点与连续重复字压缩；"
        f"整理空白与断行。不改写措辞、不纠错、不翻译、不总结。\n\n"
        f"<text>\n{raw}\n</text>"
    )
    return [
        {"role": "system", "content": "你是文本清洗器，只输出清洗后的正文。"},
        {"role": "user", "content": user},
    ]


def run(mode: str, n: int, workers: int, out_path: Path, seed: int, min_similarity: float) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if out_path.exists():
        done = sum(1 for _ in out_path.open(encoding="utf-8"))
        print(f"发现已有 {done} 条，继续追加")

    lock = threading.Lock()
    stats = {"ok": 0, "reject": 0, "error": 0}
    rejects: list[str] = []

    def worker(index: int) -> None:
        rng = random.Random(seed * 100003 + index)
        try:
            messages = build_gen_messages(rng)
            content = call_teacher(messages, temperature=rng.uniform(0.5, 0.9))
            pair = parse_pair(content)
            if pair is None:
                with lock:
                    stats["error"] += 1
                    rejects.append("JSON 解析失败")
                return
            record, reason = validate(pair, min_similarity=min_similarity)
            with lock:
                if record is None:
                    stats["reject"] += 1
                    rejects.append(reason)
                    return
                record["id"] = hashlib.sha1(
                    (record["raw"] + record["clean"]).encode("utf-8")
                ).hexdigest()[:16]
                with out_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["ok"] += 1
                if stats["ok"] % 25 == 0:
                    print(f"  ... 已写入 {stats['ok']} 条")
        except Exception as exc:  # noqa: BLE001
            with lock:
                stats["error"] += 1
                rejects.append(f"{type(exc).__name__}: {exc}")

    from concurrent.futures import ThreadPoolExecutor

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, range(done, done + n)))

    elapsed = time.time() - start
    print(f"\n完成：成功 {stats['ok']}，拒绝 {stats['reject']}，异常 {stats['error']}，"
          f"耗时 {elapsed:.1f}s（{stats['ok'] / max(1e-9, elapsed):.2f} 条/秒）")
    if rejects:
        from collections import Counter

        print("拒绝原因 Top5：")
        for reason, cnt in Counter(r[:40] for r in rejects).most_common(5):
            print(f"  {cnt:>4}  {reason}")
    return 0 if stats["ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gen", "label"], default="gen")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "data" / "teacher" / "gen.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-similarity", type=float, default=0.90)
    args = ap.parse_args()
    if args.mode != "gen":
        raise SystemExit("label 模式由 build_teacher_pairs.py 调用，暂不单独开放")
    return run(args.mode, args.n, args.workers, Path(args.out), args.seed, args.min_similarity)


if __name__ == "__main__":
    raise SystemExit(main())
