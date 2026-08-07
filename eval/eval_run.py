"""逐条跑评测：对每条问题走完整 RAG 链路并落盘原始产物。

链路（与 webapp/server.py 完全一致）：
    问题 → 可选查询改写 → 双路召回(dense+BM25+RRF, top_k=15)
         → 可选 OpenAI-compatible 多模态模型生成带[E#]引用的答案

每条记录保留：检索式(query)+改写说明(note)、检索到的 chunks(含 rrf/dense/bm25
排名与 description)、生成答案、模型名。结果按问题集分别落两份文件。

用法（在项目根目录执行；生成答案需要配置 RAG_LLM_* 环境变量）：
    # Windows PowerShell:
    $env:RAG_CORPUS="tkd"; python eval/eval_run.py
    # bash:
    RAG_CORPUS=tkd python eval/eval_run.py
    # 只跑某一份 / 限量调试：
    RAG_CORPUS=tkd python eval/eval_run.py --only scenarios --limit 3

特性：
- 断点续传：已生成且含 gemma_answer 的条目自动跳过；可反复运行补齐。
- 增量落盘：每条完成即写文件，中途中断不丢已完成结果。
- 查询改写/答案生成失败不致命：记录 error 字段，继续后面的题。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAG_CORPUS", "tkd")           # 默认泰晶语料

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

# 复用 webapp 的 deepseek 改写 / 检索 / gemma 生成，确保与线上行为一致
from webapp.server import INDEX, deepseek_rewrite, do_search, gemma_answer  # noqa: E402

_BY_ID = {c["chunk_id"]: c for c in INDEX.chunks}

SETS = {
    "eval":      (EVAL_DIR / "seed_eval.jsonl",      EVAL_DIR / "answers_seed_eval.json"),
    "scenarios": (EVAL_DIR / "seed_scenarios.jsonl", EVAL_DIR / "answers_seed_scenarios.json"),
}


def _load_done(out_path: Path) -> dict:
    if out_path.exists():
        try:
            return {r["id"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))}
        except Exception:
            return {}
    return {}


def _full_chunks(chunk_ids: list[str]) -> list[dict]:
    """从索引取完整 chunk（gemma_answer 需要 native_text 全文 + crop_images）。"""
    chunks = []
    for cid in chunk_ids:
        c = _BY_ID.get(cid)
        if c:
            chunks.append({**c, "crop_images": [p.replace("\\", "/") for p in c["crop_images"]]})
    return chunks


def run_one(item: dict, top_k: int) -> dict:
    req = item["question"]
    rec = dict(item)                                   # 保留全部 gold 字段
    t0 = time.time()

    # ① DeepSeek 改写检索式
    try:
        rw = deepseek_rewrite(req)
        query = (rw.get("query") or "").strip() or req
        note = rw.get("note", "")
        rec["deepseek_error"] = None
    except Exception as e:
        query, note = req, ""
        rec["deepseek_error"] = f"{type(e).__name__}: {e}"

    rec["deepseek_query"] = query
    rec["deepseek_note"] = note

    # ② 双路召回 + RRF（top_k=15，与 demo 一致）
    results = do_search(query, top_k)
    rec["retrieved_chunks"] = [{
        "rank": i + 1,
        "chunk_id": r["chunk_id"],
        "rrf_score": r["rrf_score"],
        "dense_rank": r["dense_rank"],
        "bm25_rank": r["bm25_rank"],
        "dense_score": r["dense_score"],
        "bm25_score": r["bm25_score"],
        "doc_id": r["doc_id"],
        "page": r["page"],
        "block_type": r["block_type"],
        "section_title": r["section_title"],
        "toc_path": r["toc_path"],
        "description": r["description"],
    } for i, r in enumerate(results)]
    retrieved_ids = [r["chunk_id"] for r in results]
    rec["retrieved_chunk_ids"] = retrieved_ids

    # 召回命中率（对照 gold_chunks，便于后续检索评估）
    gold = set(item.get("gold_chunks") or [])
    if gold:
        hit = gold & set(retrieved_ids)
        rec["retrieval_recall"] = round(len(hit) / len(gold), 3)
        rec["retrieval_hits"] = sorted(hit)

    # ③ 可选多模态模型读证据(文本+裁剪图)生成答案
    try:
        chunks = _full_chunks(retrieved_ids)
        rec["gemma_answer"] = gemma_answer(req, query, chunks)
        rec["gemma_model"] = config.LLM_MODEL
        rec["gemma_error"] = None
    except Exception as e:
        rec["gemma_answer"] = None
        rec["gemma_model"] = config.LLM_MODEL
        rec["gemma_error"] = f"{type(e).__name__}: {e}"

    rec["elapsed_sec"] = round(time.time() - t0, 1)
    return rec


def run_set(name: str, top_k: int, limit: int | None):
    in_path, out_path = SETS[name]
    lines = in_path.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    done = _load_done(out_path)
    out = [done[it["id"]] for it in items if it["id"] in done]  # 维持顺序
    out_ids = {r["id"] for r in out}
    n_new = 0
    for it in items:
        previous = done.get(it["id"], {})
        if previous.get("gemma_answer") and not previous.get("gemma_error"):
            continue                                   # 已完成，跳过
        if limit is not None and n_new >= limit:
            break
        print(f"[{name}] {it['id']} ... ", end="", flush=True)
        rec = run_one(it, top_k)
        # 替换或追加
        out = [r for r in out if r["id"] != it["id"]]
        out.append(rec)
        out_ids.add(it["id"])
        # 维持原始顺序
        order = {x["id"]: i for i, x in enumerate(items)}
        out.sort(key=lambda r: order.get(r["id"], 1e9))
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        n_new += 1
        rr = rec.get("retrieval_recall")
        generation = (
            "OK"
            if rec.get("gemma_answer")
            else "ERR:" + str(rec.get("gemma_error"))
        )
        print(f"recall={rr} generation={generation} ({rec['elapsed_sec']}s)")
    print(f"[{name}] done. total={len(out)} written -> {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["eval", "scenarios"], help="只跑其中一份")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--limit", type=int, default=None, help="本次最多新跑多少条（调试用）")
    args = ap.parse_args()

    print(f"[init] corpus={config.CORPUS} chunks={len(INDEX.chunks)} "
          f"top_k={args.top_k} llm={config.LLM_BASE_URL or 'disabled'}")
    print("[init] 预热 embedding ...")
    INDEX.embed_query("warm up")

    names = [args.only] if args.only else ["eval", "scenarios"]
    for name in names:
        run_set(name, args.top_k, args.limit)


if __name__ == "__main__":
    main()
