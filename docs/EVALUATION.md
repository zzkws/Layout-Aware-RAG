# Evaluation Protocol

**No quantitative evaluation has been run.** This document specifies the protocol
that *would* produce one, so that the claims in [METHOD.md](METHOD.md) can be tested
rather than asserted. It contains no results, and none should be inferred from it.

What the current release demonstrates is design, implementation, and qualitative
behavior on two real corpora. It does not claim benchmark accuracy, state-of-the-art
performance, or superiority over pixel-based document retrieval or any other system.

Publishing the protocol before the results is deliberate. It fixes the metrics, the
query strata, and the ablation set **in advance**, which is what prevents a favourable
subset from being selected after the fact. A protocol written after seeing the numbers
is not a protocol; it is a description of the numbers.

---

## 1. Task definition

The system under test is an **evidence retrieval** system, not a question-answering
system. Answer generation is optional in the implementation and out of scope here.

Given a natural-language or part-number query, the system returns a ranked list of
evidence packages. A package is *correct* when it points at a page region that
actually contains the information the query asks for — not when a downstream model
produces an acceptable sentence from it.

This separation is the point. Scoring the generated answer would make every metric a
joint measurement of retrieval and generation, and the design claims in
[METHOD.md](METHOD.md) are all claims about retrieval.

---

## 2. Corpora

| Corpus | Documents | Pages | Page-native chunks |
| :--- | ---: | ---: | ---: |
| TXC | 108 | 226 | 875 |
| TKD | 74 | 112 | 547 |

Both are frozen at release
[`v0.1.0`](https://github.com/zzkws/Layout-Aware-Retrieval/releases/tag/v0.1.0), the
only release carrying corpus snapshot assets. Any published result must name the
snapshot version, because retrieval scores are not comparable across corpus revisions
— and, per [METHOD.md](METHOD.md) §6, fused scores are not comparable across corpora
of different sizes either.

---

## 3. Query set construction

Target: **150 queries per corpus**, stratified as below. The strata are chosen to
separate the failure modes the design claims to address, not to be representative of
any measured query distribution — no such distribution has been collected, and the
proportions are a design decision that should be stated as one.

| Stratum | Share | Example |
| :--- | ---: | :--- |
| Exact part number | 25% | `7M-26.000MAAJ-T` |
| Partial / remembered part number | 15% | `26.000M crystal` |
| Parametric lookup | 25% | `32.768 kHz load capacitance` |
| Table-resident condition | 15% | `frequency tolerance at −40 °C` |
| Figure / drawing reference | 10% | `land pattern dimensions for 3225 package` |
| Cross-page or multi-document | 10% | `which parts support 1.8 V supply` |

Two constraints on authoring:

1. **Queries must be written before any retrieval run.** Otherwise the query set
   drifts toward what the system already answers well.
2. **Queries must be written from the source PDFs, not from the chunk index.**
   Authoring from the index biases phrasing toward the segmentation the system happens
   to produce, which is the specific thing under test.

Both English and Chinese phrasings should appear, since the embedding model is
multilingual and the corpora are not uniformly one language.

---

## 4. Ground truth

For each query, annotate the set of page regions that answer it:

```
query_id · doc_id · page · bbox_pdf[] · relevance ∈ {2, 1, 0}
```

`2` = fully answers the query; `1` = partially relevant, or supplies context the
answer requires; `0` = explicitly judged non-relevant (a hard negative, retained
rather than discarded so that a later run can distinguish "judged irrelevant" from
"never judged").

**Annotation is at region level, not chunk level.** This is the central
methodological point of the protocol. Annotating chunks would make the system's own
segmentation its ground truth, and the page-native design — the thing being evaluated
— would become untestable by construction. Region annotations are matched to retrieved
chunks afterwards by geometric overlap (§5.2).

Annotations are in `bbox_pdf`, the authoritative coordinate
([METHOD.md](METHOD.md) §2), so they remain valid if the corpus is re-rendered at a
different DPI. They should not be reported to a precision finer than the storage
carries (§2.2 of METHOD.md).

Single-annotator labelling must be declared as such. A ≥ 20% double-annotated subset
with Cohen's κ reported is the minimum for any comparative claim.

---

## 5. Metrics

### 5.1 Ranking

Recall@{1, 5, 10}, nDCG@10 (Järvelin & Kekäläinen, 2002), and MRR, computed over
graded relevance. A chunk counts as relevant for these when it meets the geometric
match criterion of §5.2.

### 5.2 Evidence quality

These are the metrics specific to this system's claims, and the ones a text-chunk
baseline cannot be scored on directly. Let `R` be the annotated region and `C` the
union of a retrieved chunk's `bboxes_pdf`.

| Metric | Definition | What it detects |
| :--- | :--- | :--- |
| **Page-hit rate@k** | fraction of queries with a retrieved chunk on the correct page within the top *k* | coarse localization |
| **Region IoU@1** | `area(R ∩ C) / area(R ∪ C)` for the top-1 chunk | geometric precision. **Report as a distribution, not a mean** — the failure cases are the informative part and a mean hides them |
| **Evidence sufficiency@1** | `area(R ∩ C) / area(R) ≥ 0.95` | truncation. This is the metric that tests the area-first de-duplication rule of METHOD.md §3 |
| **Context leakage@1** | `area(C) / area(R)` | over-return |

**Sufficiency and leakage must be reported as a pair.** Reporting sufficiency alone
admits a degenerate solution: a system that returns the whole page scores perfectly on
sufficiency and has learned nothing. Leakage is what makes that visible. A result
citing one without the other should be treated as incomplete.

### 5.3 Cost

Index build time, index size on disk, and query latency (p50 / p95), CPU-only, stated
with the hardware. Index size should be reported with and without the character
3-grams of METHOD.md §5.1, since that is the component with the clearest size cost.

---

## 6. Ablations

Each row isolates one design decision from [METHOD.md](METHOD.md). Without these, the
design rationales remain untested arguments.

| # | Ablation | Tests |
| :--- | :--- | :--- |
| **A1** | dense only | §6 fusion |
| **A2** | BM25 only | §6 fusion |
| **A3** | RRF fusion — the full system | §6 |
| **A4** | standard tokenizer vs part-aware | §5.1 |
| **A5** | part-aware without character 3-grams | §5.1 |
| **A6** | `embed_text` without `description` | §5 |
| **A7** | `embed_text` without `toc_path` + `section_title` | §5 |
| **A8** | de-duplication by confidence vs by area | §3 |
| **A9** | `RRF_K` ∈ {10, 30, 60, 100} | §6 |

**A8 is the ablation that matters most**, and it has a stated prediction: it should
move *evidence sufficiency* while leaving ranking metrics roughly unchanged. That is
the whole argument for the rule — it claims to protect evidence completeness, not
ranking quality. If A8 moves ranking metrics instead, or moves nothing, the rule is
not earning its place and should be reconsidered rather than defended.

A9 is cheap by construction: BM25 and RRF scoring both happen at query time, so the
sweep needs no index rebuild ([METHOD.md](METHOD.md) §5.2).

---

## 7. Baselines

| # | Baseline | Purpose |
| :--- | :--- | :--- |
| **B1** | Fixed-size text chunking (512 / 1024 tokens, 10% overlap) over the same PDFs | the design's actual alternative |
| **B2** | Page-level retrieval, whole page as the unit | ceiling on page-hit, floor on leakage |
| **B3** | BM25 over raw extracted text, no layout | cost-free lower bound |

B1 and B3 cannot be scored on region IoU or context leakage in the same way, because
they carry no region geometry. They should be scored on page-hit and ranking metrics,
with the geometric columns marked not-applicable rather than left blank — the
inapplicability is itself part of the argument for the design (claim C3 in the
[README](../README.md#evaluation-status)).

A page-image (pixel) RAG comparison is **not** in scope. It would require matching a
generation model and a visual encoder across systems, which introduces more confounds
than the comparison resolves. Until such a study is designed properly, the README's
statement that no superiority over pixel-based retrieval is claimed stands as written.

---

## 8. Threats to validity

- **Single annotator, single author.** Query authoring, annotation, and system design
  share one person. Expectancy bias is not controlled for, and no blinding is possible
  in this setup. This is the most serious threat on the list.
- **Two corpora, one domain.** Both are crystal / oscillator datasheets from adjacent
  vendors. Nothing here generalizes to other document families without being retested.
- **Born-digital only.** Scanned PDFs are outside the current pipeline (no OCR
  engine), so the query set cannot include them, and the protocol says nothing about
  how the system behaves on them.
- **Offline VLM enrichment is frozen.** Descriptions and TOC paths were produced by
  one model at one point in time. Regenerating them changes the index, so a result is
  a measurement of one specific index, not of the pipeline in general.
- **Corpus size.** At 875 and 547 chunks, confidence intervals on Recall@1 will be
  wide. Report them; do not report bare point estimates.
- **Query-set size.** 150 queries per corpus, stratified six ways, leaves 15 queries
  in the smallest stratum. Per-stratum results from a set that size are indicative at
  best and should be labelled as such.

---

## 9. Reporting rules

Any published number must state:

1. corpus snapshot version,
2. commit SHA,
3. embedding model and revision,
4. hardware,
5. query set version.

Results lacking these are not reproducible and should not be cited from this
repository.

---

## References

- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales.* Educational and
  Psychological Measurement, 20(1), 37–46.
- Järvelin, K., & Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR
  Techniques.* ACM Transactions on Information Systems, 20(4), 422–446.
