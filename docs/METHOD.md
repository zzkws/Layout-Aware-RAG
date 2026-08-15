# Method / 方法

This document records the design decisions behind the pipeline and the reasoning
for each one. It describes what the code actually does; parameter values are
those in [`config.py`](../config.py).

本文记录管线的设计决策及其取舍理由，描述的是代码的实际行为；参数取值以
[`config.py`](../config.py) 为准。

## 1. Problem statement / 问题定义

Engineering datasheets are layout-bearing documents. A load-capacitance value is
meaningful because of the table row it sits in, the part-number column heading
above it, and the note beneath the table. Flattening such a page into
fixed-length token windows destroys exactly the structure that carries the
meaning, and it destroys the reader's ability to check the answer.

工程数据手册是**版面承载语义**的文档。一个负载电容值之所以有意义，是因为它所在
的表格行、上方的型号列头，以及表格下方的注释。把这样的页面压平成定长 token 窗口，
恰好破坏了承载语义的结构，也破坏了读者复核答案的能力。

The design target is therefore not "better answers" but **an inspectable
retrieval unit**: every result must be able to return to its document, page,
PDF coordinates, native text, and section path.

因此设计目标不是"更好的回答"，而是**可检查的检索单元**：每个结果都必须能回到它的
文档、页码、PDF 坐标、原文和章节路径。

## 2. Coordinate contract / 坐标契约

Two coordinate systems are maintained per element:

- `bbox_pdf` — PDF points (72 dpi basis). **Authoritative**, independent of
  render resolution.
- `bbox_px` — pixel coordinates of the rendered page, used only for cropping and
  visualization. `bbox_px = bbox_pdf * dpi / 72`.

每个元素维护两套坐标：`bbox_pdf`（PDF 点，72 dpi 基准）是**权威坐标**，与渲染
分辨率无关；`bbox_px`（渲染图像素坐标）仅用于裁图和可视化。

Pages render at `RENDER_DPI = 200`. Because the authoritative coordinate is
resolution-independent, the render DPI can be raised for better layout detection
or lowered for speed without invalidating any stored evidence.

页面按 `RENDER_DPI = 200` 渲染。由于权威坐标与分辨率无关，渲染 DPI 可以为了更好的
版面检测而调高、或为了速度而调低，都不会让已存储的证据失效。

## 3. Layout detection and de-duplication / 版面检测与去重

Detection uses DocLayout-YOLO (`DocStructBench` weights, `imgsz = 1024`,
`conf = 0.25`) over the 10-class DocStructBench label set.

The raw detector output contains two kinds of duplicates: co-located double
boxes on a single element, and same-class nesting where a large region also
yields a smaller box inside it. De-duplication is a greedy pass sorted by
**area descending, then confidence descending**, with three drop rules:

| Condition | Rule |
|---|---|
| Any class, `IoU > 0.85` | keep the larger box |
| Same class, `IoU > 0.55` | keep the larger box |
| Same class, ≥85% of the smaller box lies inside the larger | keep the larger box |

**Why area rather than confidence.** YOLO frequently assigns *lower* confidence
to the box that contains more content. In document `6u`, the `table_footnote`
region yields a complete box at confidence 0.37 and a truncated box at 0.73;
ranking by confidence would silently cut off the beginning of the Note line.
Ranking by area keeps the box that preserves more of the source.

**为什么按面积而不是置信度。** YOLO 经常给内容更完整的框*更低*的置信度。在文档
`6u` 中，`table_footnote` 区域的完整框置信度为 0.37、截断框为 0.73；按置信度保留
会悄悄裁掉 Note 行的开头。按面积保留，则保住了信息更完整的那个框。

**Cross-class nesting is deliberately preserved.** A `plain_text` label inside a
`figure` is usually real annotation content, not a duplicate. Resolving it is a
grouping decision, not a detection decision, so it is deferred to stage 4.

**跨类别包含关系刻意保留。** `figure` 内部的 `plain_text` 通常是真实的标注文字而
非重复框。它属于分组决策而非检测决策，因此推迟到阶段 4 处理。

## 4. Page-native grouping / 版面级分组

A chunk is not a token window; it is a group of layout elements that a human
would read as one unit — a table with its caption and footnote, a figure with
its annotations. Grouping, section titles, and TOC paths are produced offline by
a VLM pass (`GEMINI_MODEL`, default `gemini-2.5-flash`) and then frozen into the
index. The VLM is an *offline enrichment step*, not a query-time dependency:
retrieval runs with no API key and no cloud call.

Chunk 不是 token 窗口，而是**人类会当作一个单元来读**的一组版面元素——表格连同它的
标题和脚注、图连同它的标注。分组、章节标题和目录路径由离线 VLM 阶段生成后冻结进
索引。VLM 是*离线富化步骤*，不是查询期依赖：检索本身不需要 API Key，也不产生云端调用。

## 5. Dual-path indexing / 双路索引

The two retrieval paths deliberately index **different text compositions**,
because they fail in different ways.

两条检索路径刻意索引**不同的文本组合**，因为它们的失效模式不同。

**Dense path** — `embed_text` concatenates, in this order:

```
toc_path → section_title → description → native_text
```

truncated to 1500 characters. The ordering encodes a hierarchy: the TOC path
says *where* the content sits, the section title says *what* it is, and only
then come the semantic summary and the raw text. Front-loading location and
identity means truncation degrades the tail (raw text) rather than the context.

按 `toc_path → section_title → description → native_text` 顺序拼接后截断至 1500
字符。这个顺序编码了一个层级：目录路径说明内容*在哪*，章节标题说明它*是什么*，
之后才是语义概述和原文。把位置与身份前置，意味着截断损失的是尾部原文而非上下文。

Embeddings come from `microsoft/harrier-oss-v1-0.6b` (multilingual, 1024-dim)
via sentence-transformers, using the model's built-in `web_search_query` prompt
for queries. A lighter `fastembed` / `bge-small-en-v1.5` route exists as a
fallback backend.

**Sparse path** — `fts_text` is a wider bag: section title, TOC path, native
text, description, keywords, and document-card words. Recall matters more than
precision here, since RRF will arbitrate.

稀疏路的 `fts_text` 是更宽的词袋（章节标题、目录路径、原文、描述、关键词、文档卡
词）。这一路召回比精度更重要，因为最终由 RRF 仲裁。

### 5.1 Model-number-aware tokenization / 型号感知分词

This is the single most datasheet-specific decision in the system. A standard
tokenizer splits `7M-26.000MAAJ-T` into `7 / M / 26 / 000 / MAAJ / T`, which
destroys part-number search — the dominant query type for this corpus.

这是全系统最贴合数据手册场景的一个决策。标准分词器会把 `7M-26.000MAAJ-T` 拆成
`7 / M / 26 / 000 / MAAJ / T`，型号检索随之失效——而型号正是该语料最主要的查询类型。

A token is treated as part-like when it matches
`^(?=.*\d)[a-z0-9][a-z0-9.\-_/±]{4,}$` — long, contains a digit, contains
separators. Part-like tokens are indexed **three ways**:

| Channel | Purpose |
|---|---|
| the intact token | exact match |
| alphanumeric subfields | partial recall (user remembers part of it) |
| character 3-grams (separators stripped) | substring match (`26.000M` alone) |

Ordinary words take the normal lowercase path. See
[`common/tokenize.py`](../common/tokenize.py).

### 5.2 BM25

Textbook Okapi BM25 with `K1 = 1.5`, `B = 0.75`, and

```
idf = ln(1 + (N − df + 0.5) / (df + 0.5))
```

The inverted index and document lengths are precomputed; scoring happens at
query time so that ranking parameters can be changed without a rebuild.

倒排索引与文档长度预先构建，打分在查询期进行，因此调整排序参数无需重建索引。

## 6. Rank fusion / 排序融合

Fusion is Reciprocal Rank Fusion with `RRF_K = 60`, over the **full** ranked
list from each path:

```
score(d) = Σ_paths  1 / (RRF_K + rank_path(d) + 1)
```

RRF is chosen over score-weighted fusion because dense cosine similarity and
BM25 scores are not on a comparable scale, and BM25 scores in particular vary
with query length and corpus statistics. Using ranks makes fusion invariant to
both, which means the embedding model can be swapped without retuning a fusion
weight.

选择 RRF 而非分数加权融合，是因为 dense 余弦相似度与 BM25 分数不在可比量纲上，
且 BM25 分数还随查询长度和语料统计量变化。改用**排名**使融合对二者都不敏感——
这意味着更换 Embedding 模型时不需要重新调融合权重。

Results retain `dense_rank`, `bm25_rank`, `dense_score`, and `bm25_score`
alongside the fused score, so any ranking can be attributed to a path after the
fact. `TOP_K = 15` by default.

结果同时保留 `dense_rank`、`bm25_rank`、`dense_score`、`bm25_score`，因此任何排序
结果事后都可归因到具体路径。

## 7. The evidence package / 证据包

The output contract is the evidence package, not a model response:

```
chunk_id · doc_id · source_pdf · page · bboxes_pdf
block_type · section_title · toc_path
native_text · description · crop_images
```

Every field is inspectable, and every field survives replacement of the layout
model, the embedding model, the fusion strategy, or the answer model. This is
the reason the same contract can be served over three surfaces — the local
Python pipeline, the static demo sites, and the MCP server — without
re-implementation.

每个字段都可检查，且在版面模型、Embedding 模型、融合策略或生成模型被替换后依然
成立。这也是同一份契约能在本地 Python 管线、静态 Demo 站和 MCP 服务器三个界面上
复用而无需重新实现的原因。

## 8. Known limitations / 已知边界

See [Current boundaries](../README.md#current-boundaries--当前边界) in the
README, and [EVALUATION.md](EVALUATION.md) for what has and has not been
measured.

本项目**尚未进行定量测评**。第 3、5、6 节中的设计理由是工程论证与个案观察，
不是实验结论。评测协议见 [EVALUATION.md](EVALUATION.md)。

The rationales given in sections 3, 5, and 6 are engineering arguments supported
by individual observed cases — they are not experimental findings.
