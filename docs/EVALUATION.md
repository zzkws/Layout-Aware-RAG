# Evaluation / 评测

## Status / 当前状态

**No quantitative evaluation has been run.** This document specifies the
protocol that *would* produce one, so that the claims in
[METHOD.md](METHOD.md) can be tested rather than asserted. It contains no
results, and none should be inferred from it.

**本项目尚未进行定量测评。** 本文给出的是*可用于产生测评结果*的协议，目的是让
[METHOD.md](METHOD.md) 中的论断可被检验而非仅被声明。本文不含任何结果，也不应从中
推断出任何结果。

What the current release demonstrates is design, implementation, and qualitative
behavior on two real corpora. It does not claim benchmark accuracy, SOTA
performance, or superiority over pixel-based RAG or any other system.

Publishing the protocol before the results is deliberate: it fixes the metrics
and the ablation set in advance, which is what prevents a favourable subset from
being selected after the fact.

先公布协议、后产出结果是刻意的：它把指标和消融集合**提前固定**，从而避免事后
挑选有利子集。

## 1. Task definition / 任务定义

The system under test is an **evidence retrieval** system, not a question
answering system. Answer generation is optional and out of scope here.

被测系统是**证据检索**系统，而非问答系统。答案生成是可选项，不在本协议范围内。

Given a natural-language or part-number query, the system returns a ranked list
of evidence packages. A package is *correct* when it points at a page region
that actually contains the information the query asks for.

## 2. Corpora / 语料

| Corpus | Documents | Pages | Page-native chunks |
|---|---:|---:|---:|
| TXC | 108 | 226 | 875 |
| TKD | 74 | 112 | 547 |

Both are frozen at release [`v0.1.0`](https://github.com/zzkws/evidence-rag-pilot/releases/tag/v0.1.0),
which is the only release carrying corpus snapshot assets. Any published result
must name the snapshot version, because retrieval scores are not comparable
across corpus revisions.

两个语料均冻结在 `v0.1.0`（唯一携带语料快照资产的 release）。任何公布的结果都必须
标明快照版本，因为跨语料修订版的检索分数不可比。

## 3. Query set construction / 查询集构建

Target: **150 queries per corpus**, stratified as follows. The strata are chosen
to separate the failure modes the design claims to address.

| Stratum | Share | Example |
|---|---:|---|
| Exact part number | 25% | `7M-26.000MAAJ-T` |
| Partial / remembered part number | 15% | `26.000M crystal` |
| Parametric lookup | 25% | `32.768 kHz load capacitance` |
| Table-resident condition | 15% | `frequency tolerance at −40 °C` |
| Figure / drawing reference | 10% | `land pattern dimensions for 3225 package` |
| Cross-page or multi-document | 10% | `which parts support 1.8 V supply` |

Queries must be written **before** any retrieval run, from the source PDFs
rather than from the chunk index, so that query phrasing is not biased toward
the segmentation the system happens to produce. Both English and Chinese
phrasings should appear, since the embedding model is multilingual and the
corpus is not.

查询必须在任何检索运行**之前**、依据源 PDF（而非 chunk 索引）撰写，以免查询措辞
偏向系统恰好产生的切分方式。

## 4. Ground truth / 标注

For each query, annotate the set of page regions that answer it:

```
query_id · doc_id · page · bbox_pdf[] · relevance ∈ {2, 1, 0}
```

`2` = fully answers the query; `1` = partially relevant or provides required
context; `0` = explicitly judged non-relevant (hard negative).

Annotation is at **region** level, not chunk level. This is the important
methodological point: annotating chunks would make the segmentation its own
ground truth and render the page-native design untestable. Region annotations
are then matched to retrieved chunks by geometric overlap (§5.2).

标注在**区域**级而非 chunk 级。这是关键的方法论要点：按 chunk 标注会让切分方式
成为自己的标准答案，使版面级设计变得不可检验。

Single-annotator labelling must be declared as such. A ≥20% double-annotated
subset with Cohen's κ reported is the minimum for any comparative claim.

单标注者标注必须如实声明。任何比较性论断至少需要 ≥20% 的双标注子集并报告
Cohen's κ。

## 5. Metrics / 指标

### 5.1 Ranking

Recall@{1,5,10}, nDCG@10, MRR — computed over graded relevance.

### 5.2 Evidence quality / 证据质量

These are the metrics specific to this system's claims, and the ones a
text-chunk baseline cannot be scored on directly:

- **Page-hit rate@k** — the retrieved chunk is on the correct page.
- **Region IoU@1** — geometric IoU between the top chunk's `bboxes_pdf` union
  and the annotated region. Reported as a distribution, not a mean.
- **Evidence sufficiency@1** — the annotated region is ≥95% contained in the
  returned chunk. This measures the truncation failure that §3 of METHOD.md
  claims the area-first de-duplication rule prevents.
- **Context leakage@1** — returned area ÷ annotated area. High sufficiency with
  high leakage means the system is winning by returning the whole page, and must
  be reported alongside sufficiency to make that visible.

`Evidence sufficiency` 与 `context leakage` 必须成对报告：只报前者的话，"整页返回"
这种退化解会被误判为高分。

### 5.3 Cost

Index build time, index size, and query latency (p50 / p95), CPU-only, stated
with the hardware.

## 6. Ablations / 消融

Each row isolates one design decision from METHOD.md. Without these, the design
rationales remain untested.

| # | Ablation | Tests the claim in |
|---|---|---|
| A1 | dense only | §6 fusion |
| A2 | BM25 only | §6 fusion |
| A3 | RRF fusion (full system) | §6 |
| A4 | standard tokenizer vs part-aware | §5.1 |
| A5 | part-aware without char 3-grams | §5.1 |
| A6 | `embed_text` without `description` | §5 |
| A7 | `embed_text` without `toc_path` + `section_title` | §5 |
| A8 | de-duplication by confidence vs by area | §3 |
| A9 | `RRF_K` ∈ {10, 30, 60, 100} | §6 |

A8 is the one that matters most for the evidence-quality metrics: it should move
`evidence sufficiency` while leaving ranking metrics roughly unchanged. If it
does not, the rule is not earning its place.

## 7. Baselines / 基线

| Baseline | Purpose |
|---|---|
| Fixed-size text chunking (512 / 1024 tokens, 10% overlap) over the same PDFs | the design's actual alternative |
| Page-level retrieval (whole page as the unit) | ceiling on page-hit, floor on leakage |
| BM25 over raw extracted text, no layout | cost-free lower bound |

A page-image (pixel) RAG comparison is **not** in scope. It would require
matching a generation model and a visual encoder, which introduces more
confounds than the comparison resolves; the README's no-superiority statement
regarding Pixel RAG stands until such a study is designed properly.

Pixel RAG 的对比**不在**本协议范围：这类比较需要匹配生成模型与视觉编码器，引入的
混杂因素比它能解决的问题更多。在此类研究被恰当设计之前，README 中"不声明优于
Pixel RAG"的表述继续成立。

## 8. Threats to validity / 效度威胁

- **Single annotator, single author.** Query authoring, annotation, and system
  design share one person; expectancy bias is not controlled for.
- **Two corpora, one domain.** Both are crystal/oscillator datasheets from
  adjacent vendors. Nothing here generalizes to other document families.
- **Born-digital only.** Scanned PDFs are outside the current pipeline (no OCR
  engine), so the query set cannot include them.
- **Offline VLM enrichment is frozen.** Descriptions and TOC paths were produced
  by one model at one point in time; regenerating them changes the index.
- **Corpus size.** At 875 and 547 chunks, confidence intervals on Recall@1 will
  be wide. Report them; do not report bare point estimates.

## 9. Reporting rules / 报告规则

Any published number must state: corpus snapshot version, commit SHA,
embedding model and revision, hardware, and query set version. Results without
these are not reproducible and should not be cited from this repository.

任何公布的数字都必须同时标明：语料快照版本、commit SHA、Embedding 模型及其修订、
硬件、查询集版本。缺少这些的结果不可复现，不应从本仓库引用。
