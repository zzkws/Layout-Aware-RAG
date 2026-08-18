# Method

This document records the design decisions behind the pipeline and the reasoning for
each. It describes what the code actually does; every parameter value quoted here is
the value in [`config.py`](../config.py) or in the module named beside it.

The [README](../README.md) states these decisions at the level needed to understand
the system. This document adds the parts that only matter once you intend to modify,
extend, or evaluate it: the per-stage artifact contracts, the validation guards, and
the places where a stated invariant holds only approximately.

---

## 1. Problem statement

Engineering datasheets are **layout-bearing** documents. A load-capacitance value is
meaningful because of the table row it occupies, the part-number column heading above
it, and the note beneath the table. Flattening such a page into fixed-length token
windows destroys exactly the structure that carries the meaning, and it destroys the
reader's ability to check the answer.

The design target is therefore not "better answers" but **an inspectable retrieval
unit**: every result must be able to return to its document, page, PDF coordinates,
native text, and section path.

This reframes what the system is optimizing. A conventional RAG chunker optimizes for
a downstream generation model's context window. This pipeline optimizes for a human
or agent who intends to *check* the result, which produces different decisions at
almost every stage — most visibly in §3, where a rule that slightly hurts detection
precision is preferred because it never truncates evidence.

---

## 2. Coordinate contract

### 2.1 Two systems, one authority

Two coordinate systems are maintained per element:

| Field | Basis | Role |
| :--- | :--- | :--- |
| `bbox_pdf` | PDF points, 72 dpi basis | **Authoritative.** Stored, published, and used for all downstream reasoning. |
| `bbox_px` | rendered-page pixels | Derived and disposable. Cropping and visualization only. |

with `bbox_px = bbox_pdf × dpi / 72`. Pages render at `RENDER_DPI = 200`.

Because the authoritative coordinate is resolution-independent, the render DPI can be
raised for better layout detection or lowered for speed **without invalidating any
stored evidence**. Re-rendering at a different DPI changes every `bbox_px` in the
corpus and no `bbox_pdf`.

### 2.2 Where the precision actually comes from

The invariant above is a storage and interface property, not a claim about
measurement. Detection runs on the *rendered image*, so the box is discovered in
pixel space and converted once, in [`pipeline/layout.py`](../pipeline/layout.py):

```
scale       = 72.0 / render_dpi
bbox_pdf    = round(bbox_px × scale, 2)
```

Two consequences follow, and both matter to anyone evaluating the system:

1. **Detection precision is bounded by render DPI.** At 200 dpi one pixel is 0.36 pt.
   Raising the DPI does not merely re-derive the same boxes at finer granularity; it
   can change which boxes the detector finds at all. An index built at a different
   `RENDER_DPI` is therefore a different index, even though the coordinate contract
   is preserved.
2. **`bbox_pdf` is rounded to 0.01 pt**, and the pixel box it derives from is itself
   rounded to 0.1 px. Region-IoU figures computed against manual annotations should
   not be reported to a precision the storage does not carry.

### 2.3 Crops are padded; stored boxes are not

Evidence crops are cut with `CROP_PAD = 8` px of margin on each side, clipped to the
page ([`pipeline/vlm_blocks.py`](../pipeline/vlm_blocks.py)). The stored `bboxes_px`
and `bboxes_pdf` are the **unpadded** detector boxes.

The padding exists because a box cut exactly at the glyph boundary is unpleasant to
read and occasionally clips descenders. The asymmetry is deliberate but must be
declared: the evidence *image* covers marginally more of the page than the evidence
*coordinates* claim. At 200 dpi the margin is 2.88 pt per side. Any metric computed
from coordinates is unaffected; any metric computed by inspecting the image is
measuring a slightly larger region than the record describes.

---

## 3. Layout detection and de-duplication

### 3.1 Detector

DocLayout-YOLO (Zhao et al., 2024), `DocStructBench` weights, `imgsz = 1024`,
`conf = 0.25`, over the 10-class DocStructBench label set: `title`, `plain_text`,
`abandon`, `figure`, `figure_caption`, `table`, `table_caption`, `table_footnote`,
`isolate_formula`, `formula_caption`. Weights are fetched from Hugging Face, with an
automatic fallback to `hf-mirror.com` when the direct connection fails.

### 3.2 Two kinds of duplicate

The raw detector output contains:

- **co-located double boxes** — one element yielding two near-identical boxes;
- **same-class nesting** — a large region that also yields a smaller box inside it.

### 3.3 The rule

De-duplication is a greedy pass over elements sorted by **area descending, then
confidence descending**. Each candidate is dropped if, against any already-kept box:

| Condition | Rule |
| :--- | :--- |
| Any class, `IoU > 0.85` | keep the larger box |
| Same class, `IoU > 0.55` | keep the larger box |
| Same class, ≥ 85% of the smaller box lies inside the larger | keep the larger box |

Because the sort is area-descending, a candidate can never be larger than a box
already kept, so "drop the candidate" and "keep the larger box" are the same
operation. The implementation is in `dedup_elements`,
[`pipeline/layout.py`](../pipeline/layout.py); per-page counts `n_raw_detections` and
`n_deduped` are persisted so the rule's effect is auditable per document.

### 3.4 Why area rather than confidence

This is the single rule most likely to be read as arbitrary, and it is the one that
most directly serves the design target.

YOLO frequently assigns *lower* confidence to the box that contains more content. In
document `6u`, the `table_footnote` region yields a complete box at confidence 0.37
and a truncated box at 0.73. Ranking by confidence keeps the truncated box and
silently cuts off the beginning of the Note line.

The failure has three properties that make it worth a dedicated rule:

- it raises no error and logs no warning;
- the truncated evidence still looks well-formed, so a reader checking the result
  sees a plausible crop and has no signal that anything is missing;
- the missing content is disproportionately *qualifying* text — notes, conditions,
  exceptions — which is exactly what a datasheet reader is checking for.

Ranking by area keeps the box that preserves more of the source. The cost is that
occasionally a slightly over-large box survives where a tighter one existed, which
inflates `context leakage` (see [EVALUATION.md](EVALUATION.md) §5.2). That trade is
accepted deliberately, and ablation **A8** exists to test whether it was worth it.

### 3.5 What is deliberately *not* de-duplicated

**Cross-class nesting is preserved.** A `plain_text` box inside a `figure` is usually
real annotation content — a dimension callout on an engineering drawing, a legend
inside a plot — not a duplicate. Discarding it would delete text that exists nowhere
else in the document.

Deciding whether such a box is part of the figure or a separate readable unit is a
*grouping* decision, not a *detection* decision. It is therefore deferred to stage 4,
where a model that can see the page makes it with full context.

---

## 4. Page-native grouping

### 4.1 A chunk is a reading unit

A chunk is not a token window. It is the set of layout elements a human would read as
one thing: a table with its caption and footnote, a figure with its annotations.

Grouping, section titles, and TOC paths are produced by a three-pass offline stage
(`GEMINI_MODEL`, default `gemini-2.5-flash`) and then **frozen into the index**:

| Pass | Scope | Input | Produces |
| :--- | :--- | :--- | :--- |
| 1 | per document | all page images | document summary card, document name (becomes the TOC root) |
| 2 | per page | summary card, page image, element-numbered overlay, element list | element → chunk grouping; a description per chunk |
| 3 | per document | all chunks as a plain-text digest in reading order | a table of contents induced from the chunks, and a `toc_path` per chunk |

Pass 2 sees a rendered overlay with numbered element boxes alongside the plain page,
so its output is a list of element indices per group rather than free-form
coordinates — the model never has to emit geometry, and geometry it emits could never
contradict the detector.

Every prompt requires English-only output, so the index composition is
language-stable even where the source pages are not.

**The VLM is an offline enrichment step, not a query-time dependency.** Retrieval runs
with no API key, no network, and no cloud call. This is the property that makes the
demonstration sites deployable as static snapshots at all.

### 4.2 The chunk record

Each chunk written to `blocks/{doc}.json` carries:

```
block_id · doc_id · page · reading_order · block_type
section_title · toc_path · toc_redundant · indexable
bboxes_px[] · bboxes_pdf[] · render_dpi
native_text · text_source · crop_images[] · page_image
elements[]  (element_index, label, conf, bbox_px)
```

Three choices in this record are load-bearing:

1. **`bboxes_pdf` is a list, never a union.** Each member element keeps its own box
   and its own crop. A union rectangle would be cheaper and would flatter every
   region-IoU number by inflating the returned area; keeping members separate is what
   makes the evidence-quality metrics honest.
2. **`native_text` is the concatenation of member texts in element order**, not a
   re-extraction over a merged region. A merged region would sweep in text from
   whatever happens to lie between the members.
3. **`elements[]` retains the detector's own `label` and `conf`.** Any chunk can be
   traced back to the detections that produced it, including the confidences that the
   area-first rule declined to rank by.

### 4.3 Validation guards

A language model deciding groupings will occasionally return something structurally
wrong. Five guards run in code afterwards; each exists because of a failure mode with
no visible symptom.

1. **Duplicate element indices are removed** — one element assigned to two groups
   would otherwise be cropped and indexed twice.
2. **Unassigned elements are recovered as single-element chunks**, with a warning
   count printed. Dropping them would delete detected content silently.
3. **Vertically discontinuous groups are split.** If members are separated by more
   than **18% of the page height** — the archetypal failure being a running header
   grouped with a footer — the group is split into vertically contiguous sub-groups
   and the description stays with the largest. Without this, a chunk's `bboxes_pdf`
   spans the entire page and its evidence image is two unrelated strips.
4. **Chunks containing a `figure` or `table` are never dropped as index-functional.**
   An empty `toc_path` normally marks a chunk as structural furniture (headers, page
   numbers) and excludes it from the index. A graphic-bearing chunk is exempt: an
   empty path falls back to the document root. The failure this prevents is losing
   precisely the content the system exists to retrieve.
5. **Crop and overlay directories are cleared per document per run**, so a re-run
   cannot leave stale images from a previous grouping beside the new boxes.

Reading order within a page is assigned by sorting groups on their minimum `y`. This
is a heuristic and is wrong for genuinely multi-column pages; the datasheets in both
corpora are predominantly single-column with full-width tables, which is the condition
under which it holds.

---

## 5. Dual-path indexing

The two retrieval paths deliberately index **different text compositions**, because
they fail in different ways.

**Dense path.** `embed_text` concatenates, in this order, truncated to 1500
characters:

```
toc_path → section_title → description → native_text
```

The ordering encodes a hierarchy: the TOC path says *where* the content sits, the
section title says *what* it is, and only then come the semantic summary and the raw
text. Front-loading location and identity means truncation degrades the tail — raw
text — rather than the context that makes a chunk findable at all. A long table
truncated at 1500 characters remains retrievable *as that table in that section*; the
reverse ordering would produce an embedding of an anonymous run of numbers.

Embeddings come from `microsoft/harrier-oss-v1-0.6b` (multilingual, 1024-dim) via
sentence-transformers, using the model's built-in `web_search_query` prompt on the
query side and no prompt on the document side. Vectors are L2-normalized at encoding
time, so the dot product taken at query time *is* cosine similarity. A lighter
`fastembed` / `bge-small-en-v1.5` route exists as a fallback backend; both are reached
through `embed_docs` / `embed_query` in
[`common/embedder.py`](../common/embedder.py), so switching backends touches one
config value.

**Sparse path.** `fts_text` is a deliberately wider bag: section title, TOC path,
native text, description, keywords, and document-card words. Recall matters more than
precision here, because RRF arbitrates afterwards. The document card contributes
document-level vocabulary — series names, package sizes, product family — so that a
query naming a product family can reach a chunk whose own text never mentions it.

### 5.1 Model-number-aware tokenization

This is the most datasheet-specific decision in the system. A standard tokenizer
splits `7M-26.000MAAJ-T` into `7 / M / 26 / 000 / MAAJ / T`, which destroys
part-number search — the dominant query type for this corpus.

A token is treated as part-like when it matches

```
^(?=.*\d)[a-z0-9][a-z0-9.\-_/±]{4,}$
```

— long, contains a digit, contains separators. Part-like tokens are indexed through
**three channels**:

| Channel | Recovers | Example on `7m-26.000maaj-t` |
| :--- | :--- | :--- |
| the intact token | exact match | `7m-26.000maaj-t` |
| alphanumeric subfields | partial recall, when the user remembers only part | `7m`, `26`, `000maaj`, `t` |
| character 3-grams, separators stripped | substring match | `7m2`, `m26`, `26.`, … |

Ordinary words take the normal lowercase path. See
[`common/tokenize.py`](../common/tokenize.py).

The three channels are not redundant. Channel 1 alone fails when the user mistypes a
separator; channel 2 alone cannot match a query that spans a separator boundary
(`26.000M`); channel 3 alone is noisy enough that it needs the other two outranking it.
Ablations **A4** and **A5** are constructed to separate exactly these contributions.

The cost is index size and a term-frequency distortion: a part-like token inflates the
document length by roughly the number of its 3-grams, which BM25's length
normalization then partly counteracts. This is a known and unmeasured trade.

### 5.2 BM25

Textbook Okapi BM25 (Robertson & Zaragoza, 2009) with `K1 = 1.5`, `B = 0.75`, and

```
idf = ln(1 + (N − df + 0.5) / (df + 0.5))
```

The inverted index and document lengths are precomputed at build time; **scoring
happens at query time**, in [`pipeline/search.py`](../pipeline/search.py). Ranking
parameters therefore change without an index rebuild, which is what makes ablation
**A9** cheap enough to actually run.

---

## 6. Rank fusion

Fusion is Reciprocal Rank Fusion (Cormack et al., 2009) with `RRF_K = 60`, over the
**full** ranked list from each path:

```
score(d) = Σ_paths  1 / (RRF_K + rank_path(d) + 1)
```

RRF is chosen over score-weighted fusion because dense cosine similarity and BM25
scores are not on a comparable scale, and BM25 scores in particular vary with query
length and corpus statistics. Using ranks makes fusion invariant to both — which means
the embedding model can be replaced without re-tuning a fusion weight. Given that the
embedding backend is explicitly swappable (§5), a fusion strategy that required
re-tuning on every swap would make that swappability nominal.

Results retain `dense_rank`, `bm25_rank`, `dense_score`, and `bm25_score` alongside
the fused score, so any ranking can be attributed to a path after the fact. `TOP_K =
15` by default.

**A property to be aware of when reporting.** Both paths rank *every* chunk, so a
chunk with a BM25 score of zero still receives a rank and contributes fused mass. The
contribution is bounded by `1/(RRF_K + N)` and is negligible at 875 and 547 chunks,
but it is not zero, and it makes fused scores weakly dependent on corpus size in a way
raw BM25 scores are not. Fused scores from different corpora are not comparable.

---

## 7. The evidence package

The output contract is the evidence package, not a model response:

```
chunk_id · doc_id · source_pdf · page · bboxes_pdf
block_type · section_title · toc_path
native_text · description · crop_images
```

Every field is inspectable, and every field survives replacement of the layout model,
the embedding model, the fusion strategy, or the answer model. This is the reason the
same contract can be served over three surfaces — the local Python pipeline, the
static demonstration sites, and the MCP server — without re-implementation.

The MCP surface ([`evidence_mcp_server.py`](../evidence_mcp_server.py)) exposes
`build_evidence_package`, `get_chunk`, and `get_evidence_image` and nothing else. An
agent consuming it cannot observe whether retrieval is dense, sparse, fused, or
replaced entirely.

---

## 8. Stage artifacts

Each stage writes a durable artifact, and each subsequent stage reads only the
previous artifact. This is what makes the pipeline resumable
([`run_full_corpus.py`](../run_full_corpus.py) skips any document whose artifact
exists) and what makes any single stage independently replaceable.

| # | Stage | Artifact | Key fields |
| ---: | :--- | :--- | :--- |
| 1 | `render` | `pages/{doc}/pages.json` + `p{n}.png` | `width_pt`, `height_pt`, `width_px`, `height_px`, `render_dpi`, `native_text_chars`, `text_source` |
| 2 | `layout` | `layout/{doc}.json` + element overlays | per element `label`, `conf`, `bbox_px`, `bbox_pdf`; per page `n_raw_detections`, `n_deduped` |
| 3 | `extract_text` | `layout/{doc}.json`, in place | `native_text` per element |
| 4 | `vlm_blocks` | `blocks/{doc}.json`, `descriptions/{doc}.json`, crops, chunk overlays | the chunk record of §4.2; `doc_card`, per-chunk `description` and `keywords` |
| 5 | `merge_crops` | `blocks/merged/{doc}/{block_id}.png` | one evidence image per indexable chunk |
| 6 | `index_build` | `index/chunks.jsonl`, `index/dense.npy`, `index/bm25.json` | `embed_text`, `fts_text`, postings, document lengths |
| 7 | `search` | — | ranked evidence packages |

Stage 1 additionally makes the born-digital determination: a page with more than 50
characters in its text layer is marked `pdf_layer`, otherwise `needs_ocr`. The
`needs_ocr` branch is recorded and then not handled — no OCR engine is wired in, so
such a page proceeds with an empty text layer and rests entirely on the stage-4
description (§9).

---

## 9. Known limitations

This project has **not** been quantitatively evaluated. The rationales in §3, §5, and
§6 are engineering arguments supported by individual observed cases — they are not
experimental findings. The protocol that would test them is in
[EVALUATION.md](EVALUATION.md), which contains no results.

Scope limits of the current implementation:

- **No OCR engine.** `needs_ocr` pages are detected and then not handled; the pipeline
  assumes a born-digital text layer. Regions backed by embedded images or vector
  curves have empty `native_text` and rest entirely on the VLM description.
- **No cell-level table structure.** Tables are retrieved as regions, so a query
  resolving to one cell returns the whole table.
- **No cross-page table joining.** A table continued across a page break stays two
  chunks with no link between them.
- **Reading order is single-column.** Ordering groups by minimum `y` is wrong for
  genuinely multi-column pages (§4.3).
- **Grouping is frozen.** Descriptions and TOC paths came from one VLM at one point in
  time. Regenerating them changes the index, so any reported result must name the
  snapshot it was computed on.

See also [Limitations and validity](../README.md#limitations-and-validity) in the
README for the validity limits that are not implementation scope.

---

## 10. References

- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal Rank Fusion
  Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR, 758–759.
- Faysse, M., et al. (2024). *ColPali: Efficient Document Retrieval with Vision
  Language Models.* arXiv:2407.01449.
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25
  and Beyond.* Foundations and Trends in Information Retrieval, 3(4), 333–389.
- Xu, Y., Li, M., Cui, L., Huang, S., Wei, F., & Zhou, M. (2020). *LayoutLM:
  Pre-training of Text and Layout for Document Image Understanding.* KDD.
- Zhao, Z., Kang, H., Wang, B., & He, C. (2024). *DocLayout-YOLO: Enhancing Document
  Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive
  Perception.* arXiv:2410.12628.
