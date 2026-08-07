"""型号感知分词器，供 BM25/FTS 使用。

默认分词器会把 7M-26.000MAAJ-T 拆成 7/M/26/000/MAAJ/T，使型号查询失效。
本分词器对"型号样"token（含数字且含 - 或 . 的长 token）做三重索引：
  1. 保留完整原 token（精确匹配通道）
  2. 拆出字母数字子段（部分记忆匹配）
  3. 字符 3-gram（子串模糊匹配，如只记得 26.000M）
普通英文词正常小写分词。
"""
import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_PARTLIKE_RE = re.compile(r"^(?=.*\d)[a-z0-9][a-z0-9.\-_/±]{4,}$")


def _char_ngrams(token: str, n: int = 3):
    s = token.replace("-", "").replace("_", "")
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else set()


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in text.lower().split():
        raw = raw.strip(".,;:()[]{}\"'")
        if not raw:
            continue
        if _PARTLIKE_RE.match(raw):
            tokens.append(raw)                       # 完整型号 token
            tokens.extend(_WORD_RE.findall(raw))     # 子段
            tokens.extend(_char_ngrams(raw))         # 3-gram
        else:
            tokens.extend(_WORD_RE.findall(raw))
    return tokens
