"""把 eval_run.py 产出的 answers_*.json 渲染成易读的 Markdown（问题+答案）。

用法：
    python eval/render_answers.py
读取 answers_seed_eval.json / answers_seed_scenarios.json，
输出 answers_seed_eval.md / answers_seed_scenarios.md。
"""
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
SETS = [
    ("参数查询题 (12)", "answers_seed_eval.json", "answers_seed_eval.md"),
    ("场景咨询题 (30)", "answers_seed_scenarios.json", "answers_seed_scenarios.md"),
]


def render(title, data):
    L = [f"# {title}", ""]
    for r in data:
        L.append(
            f"## {r['id']}  ·  {r.get('intent', '')} / "
            f"{r.get('difficulty', '')} / {r.get('lang', '')}"
        )
        L.append("")
        L.append(f"**问题**：{r['question']}")
        L.append("")
        if r.get("deepseek_query"):
            L.append(f"**检索式 (query rewrite)**：`{r['deepseek_query']}`")
            if r.get("deepseek_note"):
                L.append(f"  ·改写说明：{r['deepseek_note']}")
            L.append("")
        rc = r.get("retrieved_chunks") or []
        if rc:
            rr = r.get("retrieval_recall")
            recall_note = f"，gold 召回率 {rr}）" if rr is not None else "）"
            L.append(f"**检索到的证据块（top {len(rc)}**" + recall_note)
            for c in rc[:8]:
                L.append(f"- E{c['rank']} `{c['chunk_id']}` rrf={c['rrf_score']} "
                         f"(dense#{c['dense_rank']}/bm25#{c['bm25_rank']}) "
                         f"| {c['block_type']} | {c.get('section_title') or '-'}")
            if len(rc) > 8:
                L.append(f"- …其余 {len(rc)-8} 块略")
            L.append("")
        L.append("**生成模型答案**：")
        L.append("")
        ans = r.get("gemma_answer")
        L.append(ans if ans else f"_(无答案；错误：{r.get('gemma_error')})_")
        L.append("")
        L.append("**对照 gold_facts**：")
        for f in (r.get("gold_facts") or []):
            L.append(f"- {f}")
        L.append("")
        L.append("---")
        L.append("")
    return "\n".join(L)


def main():
    for title, src, dst in SETS:
        p = EVAL_DIR / src
        if not p.exists():
            print(f"skip {src} (not found — run eval_run.py first)")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        (EVAL_DIR / dst).write_text(render(title, data), encoding="utf-8")
        print(f"wrote {dst} ({len(data)} items)")


if __name__ == "__main__":
    main()
