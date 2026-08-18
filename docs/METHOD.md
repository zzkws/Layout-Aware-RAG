# How it works, in detail

The [README](../README.md) covers the pipeline at the level you need to use it.
This document adds what you need to *modify* it: the composition of each index, the
guards on the grouping stage, the per-stage artifact formats, and a few places
where a stated invariant only holds approximately.

Parameter values quoted here are the values in [`config.py`](../config.py) or in the
module named beside them.

---

## 1. Coordinates

Every element carries two boxes:

| Field | Basis | Role |
| :--- | :--- | :--- |
| `bbox_pdf` | PDF points (72 dpi) | Stored and authoritative. |
| `bbox_px` | rendered-page pixels | Cropping and visualization only. |

with `bbox_px = bbox_pdf × dpi / 72`, and pages rendered at `RENDER_DPI = 200`.
Because the stored coordinate is resolution-independent, changing the render DPI
does not invalidate any stored result.

**Two caveats worth knowing before you trust that too far.**

Detection runs on the *rendered image*, so boxes are found in pixel space and
converted once, in [`pipeline/layout.py`](../pipeline/layout.py):

```
scale    = 72.0 / render_dpi
bbox_pdf = round(bbox_px * scale, 2)
```

So detection precision is bounded by render DPI, and raising the DPI does not just
re-derive the same boxes more finely — it can change which boxes the detector finds
at all. An index built at a different `RENDER_DPI` is a different index.

Also, evidence crops are cut with `CROP_PAD = 8` px of margin, but the stored boxes
are **unpadded**. The evidence *image* therefore covers slightly more of the page
than the evidence *coordinates* claim — 2.88 pt per side at 200 dpi.

---

## 2. Layout detection and de-duplication

DocLayout-YOLO, `DocStructBench` weights, `imgsz = 1024`, `conf = 0.25`, over the
10-class DocStructBench label set (`title`, `plain_text`, `abandon`, `figure`,
`figure_caption`, `table`, `table_caption`, `table_footnote`, `isolate_formula`,
`formula_caption`). Weights come from Hugging Face, falling back to `hf-mirror.com`
if the direct connection fails.

The raw output contains co-located double boxes and same-class nesting.
De-duplication is a greedy pass sorted by **area descending, then confidence
descending**, dropping a candidate when, against any already-kept box:

| Condition | Action |
| :--- | :--- |
| Any class, `IoU > 0.85` | drop the candidate |
| Same class, `IoU > 0.55` | drop the candidate |
| Same class, ≥ 85% of the candidate inside the kept box | drop the candidate |

Because the sort is area-descending, a candidate is never larger than a kept box, so
"drop the candidate" is always "keep the box covering more content". Per-page
`n_raw_detections` and `n_deduped` counts are persisted, so the rule's effect is
auditable per document.

**Why area and not confidence.** YOLO frequently gives the box with more content a
lower score — the `6u` case in the README. The failure matters because it is silent:
no error, no warning, and a truncated crop still looks well-formed, so a reader
checking the result has no signal that the start of a Note line is missing. The cost
is that an occasionally over-large box survives where a tighter one existed.

**Cross-class nesting is kept on purpose.** A `plain_text` box inside a `figure` is
usually a real dimension callout or legend, not a duplicate. Whether it belongs to
the figure is a *grouping* question, not a *detection* question, so it goes to
stage 4.

---

## 3. Grouping into chunks

A chunk is the set of layout elements a human reads as one thing. Grouping, section
titles, and TOC paths come from a three-pass offline stage (`GEMINI_MODEL`, default
`gemini-2.5-flash`) and are then frozen into the index:

| Pass | Scope | Produces |
| :--- | :--- | :--- |
| 1 | per document | document summary card and document name (the TOC root) |
| 2 | per page | element → chunk grouping, plus a description per chunk |
| 3 | per document | a TOC induced from the chunks in reading order, and a `toc_path` per chunk |

Pass 2 sees a numbered-element overlay next to the plain page, so it returns element
*indices*, never coordinates — the model cannot contradict the detector. All prompts
require English-only fields, so index composition stays language-stable even where
the source pages are not.

The VLM is an offline enrichment step, not a query-time dependency. This is what
makes the demo sites deployable as static snapshots at all.

### The chunk record

```
block_id · doc_id · page · reading_order · block_type
section_title · toc_path · toc_redundant · indexable
bboxes_px[] · bboxes_pdf[] · render_dpi
native_text · text_source · crop_images[] · page_image
elements[]  (element_index, label, conf, bbox_px)
```

Three details are load-bearing:

- **`bboxes_pdf` is never a union.** Each member keeps its own box and crop.
- **`native_text` is the members concatenated in order**, not a re-extraction over a
  merged region — which would sweep in whatever lies between the members.
- **`elements[]` keeps the detector's `label` and `conf`**, so any chunk traces back
  to the detections that produced it.

### Guards

A model deciding groupings will occasionally return something structurally wrong, and
none of these failures has a visible symptom, so five checks run in code afterwards:

1. **Duplicate element indices removed** — otherwise an element is cropped and
   indexed twice.
2. **Unassigned elements recovered** as single-element chunks, with a warning count.
   Dropping them would delete detected content silently.
3. **Vertically discontinuous groups split.** Members separated by more than **18% of
   the page height** — classically a header grouped with a footer — are split into
   contiguous sub-groups, description staying with the largest. Without this, a
   chunk spans the whole page and its evidence image is two unrelated strips.
4. **Figure- and table-bearing chunks are never dropped.** An empty `toc_path`
   normally marks structural furniture and excludes a chunk from the index; a chunk
   containing a `figure` or `table` is exempt and falls back to the document root.
   This prevents losing exactly the content the system exists to retrieve.
5. **Crop and overlay directories cleared per run**, so a re-run cannot leave stale
   images beside new boxes.

Reading order within a page is assigned by minimum `y`. That is a heuristic, and it
is wrong for genuinely multi-column pages; both corpora are predominantly
single-column with full-width tables.

---

## 4. The two indexes

The paths deliberately index **different text**, because they fail differently.

**Dense.** `embed_text` concatenates, in this order, truncated to 1500 characters:

```
toc_path → section_title → description → native_text
```

The order is the point: TOC path says *where* the content sits, section title says
*what* it is, and only then the summary and the raw text. Front-loading location and
identity means truncation eats the tail. A long table cut at 1500 characters is
still retrievable *as that table in that section*; the reverse order would embed an
anonymous run of numbers.

Embeddings: `microsoft/harrier-oss-v1-0.6b` (multilingual, 1024-dim) via
sentence-transformers, with the model's built-in `web_search_query` prompt on the
query side and none on the document side. Vectors are L2-normalized at encoding
time, so the dot product at query time *is* cosine similarity. `fastembed` /
`bge-small-en-v1.5` is a fallback backend; both go through `embed_docs` /
`embed_query` in [`common/embedder.py`](../common/embedder.py).

**Sparse.** `fts_text` is a deliberately wider bag — section title, TOC path, native
text, description, keywords, and document-card words. Recall matters more than
precision here because RRF arbitrates afterwards. The document-card words contribute
series names, package sizes, and product families, so a query naming a family can
reach a chunk whose own text never mentions it.

### Part-number tokenization

A token is part-like when it matches

```
^(?=.*\d)[a-z0-9][a-z0-9.\-_/±]{4,}$
```

— long, has a digit, has separators. Part-like tokens are indexed three ways:

| Channel | Recovers | On `7m-26.000maaj-t` |
| :--- | :--- | :--- |
| intact token | exact match | `7m-26.000maaj-t` |
| alphanumeric subfields | partial recall | `7m`, `26`, `000maaj`, `t` |
| character 3-grams, separators stripped | substring match | `7m2`, `m26`, `26.`, … |

They are not redundant: channel 1 fails on a mistyped separator, channel 2 cannot
match a query spanning a separator (`26.000M`), and channel 3 alone is noisy enough
that it needs the other two outranking it. The cost is index size, plus a term-count
distortion — a part-like token inflates document length by roughly its 3-gram count,
which BM25's length normalization partly offsets. See
[`common/tokenize.py`](../common/tokenize.py).

### BM25

Textbook Okapi BM25 with `K1 = 1.5`, `B = 0.75`, and

```
idf = ln(1 + (N - df + 0.5) / (df + 0.5))
```

Postings and document lengths are precomputed; **scoring happens at query time** in
[`pipeline/search.py`](../pipeline/search.py), so ranking parameters change without
an index rebuild.

---

## 5. Fusion

Reciprocal Rank Fusion with `RRF_K = 60`, over the full ranked list from each path:

```
score(d) = Σ_paths  1 / (RRF_K + rank_path(d) + 1)
```

Ranks rather than scores, because dense cosine similarity and BM25 are not on a
comparable scale and BM25 in particular varies with query length and corpus
statistics. Since the embedding backend is meant to be swappable, a fusion strategy
needing a re-tuned weight on every swap would make that swappability nominal.

Results keep `dense_rank`, `bm25_rank`, `dense_score`, and `bm25_score` next to the
fused score, so any ranking is attributable to a path after the fact. `TOP_K = 15`.

One property to know: both paths rank *every* chunk, so a chunk scoring zero on BM25
still gets a rank and contributes a little fused mass, bounded by `1/(RRF_K + N)`.
Negligible at 875 and 547 chunks, but not zero — fused scores from different corpora
are not comparable.

---

## 6. Stage artifacts

Each stage writes a durable artifact and reads only the previous one. That is what
makes [`run_full_corpus.py`](../run_full_corpus.py) resumable — it skips any document
whose artifact exists — and what makes a single stage independently replaceable.

| # | Stage | Artifact | Key fields |
| ---: | :--- | :--- | :--- |
| 1 | `render` | `pages/{doc}/pages.json` + `p{n}.png` | `width_pt`, `height_pt`, `width_px`, `height_px`, `render_dpi`, `native_text_chars`, `text_source` |
| 2 | `layout` | `layout/{doc}.json` + overlays | per element `label`, `conf`, `bbox_px`, `bbox_pdf`; per page `n_raw_detections`, `n_deduped` |
| 3 | `extract_text` | `layout/{doc}.json`, in place | `native_text` per element |
| 4 | `vlm_blocks` | `blocks/{doc}.json`, `descriptions/{doc}.json`, crops, overlays | the chunk record above; `doc_card`, per-chunk `description` and `keywords` |
| 5 | `merge_crops` | `blocks/merged/{doc}/{block_id}.png` | one evidence image per indexable chunk |
| 6 | `index_build` | `index/chunks.jsonl`, `index/dense.npy`, `index/bm25.json` | `embed_text`, `fts_text`, postings, document lengths |
| 7 | `search` | — | ranked evidence packages |

Stage 1 also makes the born-digital call: more than 50 characters in the text layer
marks a page `pdf_layer`, otherwise `needs_ocr`. The `needs_ocr` branch is recorded
and then not handled — there is no OCR engine wired in, so such a page proceeds with
an empty text layer and rests entirely on the stage-4 description.

---

## 7. Limitations

See [Status and limitations](../README.md#status-and-limitations) in the README, and
[EVALUATION.md](EVALUATION.md) for what has and has not been measured. Nothing here
has been quantitatively evaluated; the reasoning above is engineering argument
supported by observed cases, not experimental result.
