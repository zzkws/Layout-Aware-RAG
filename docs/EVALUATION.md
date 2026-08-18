# Evaluation status

**Nothing here has been measured.** This is a demo and a design study. The
reasoning in [METHOD.md](METHOD.md) is engineering argument supported by individual
observed cases — not experimental result. No accuracy number, no SOTA claim, and no
claim of superiority over pixel-based document RAG or anything else.

This page exists so that "not evaluated" is stated plainly rather than left for
someone to discover, and so that anyone who wants to evaluate it has a starting
point that isn't naive.

---

## If you wanted to measure it

**Score retrieval, not answers.** The system returns ranked evidence packages. If
you score a generated answer instead, every number becomes a joint measurement of
retrieval and generation, and none of the design decisions here are about
generation.

**Annotate page regions, not chunks.** This is the one methodological point that
actually matters. If you annotate chunks, the system's own segmentation becomes its
own ground truth, and the page-native design — the thing under test — is untestable
by construction. Annotate regions in `bbox_pdf`, then match them to retrieved chunks
by geometric overlap afterwards.

**Report sufficiency and leakage together.** Beyond the usual Recall@k / nDCG / MRR,
two metrics are specific to this design. Let `R` be the annotated region and `C` the
union of a retrieved chunk's boxes:

| Metric | Definition | Catches |
| :--- | :--- | :--- |
| Evidence sufficiency | `area(R ∩ C) / area(R) ≥ 0.95` | truncated evidence |
| Context leakage | `area(C) / area(R)` | returning far too much |

Sufficiency alone has a degenerate solution: a system that returns the whole page
scores perfectly and has learned nothing. Leakage is what makes that visible. Report
region IoU as a distribution rather than a mean — the failures are the informative
part.

**Pin the data.** Corpora are frozen at release
[`v0.1.0`](https://github.com/zzkws/Layout-Aware-RAG/releases/tag/v0.1.0) (TXC:
108 docs / 226 pages / 875 chunks; TKD: 74 / 112 / 547). Retrieval scores are not
comparable across corpus revisions, and per METHOD.md §5 fused scores are not
comparable across corpora of different sizes either. Any number should also state
the commit SHA, embedding model revision, and hardware.

---

## The comparisons that would be informative

| Compare against | Tells you |
| :--- | :--- |
| Fixed-size chunking (512 / 1024 tokens, 10% overlap) over the same PDFs | whether page-native chunking is worth the pipeline |
| Whole-page retrieval | the ceiling on page-hit and the floor on leakage |
| Dense only / BM25 only / RRF | whether fusion earns its place |
| Standard tokenizer vs part-aware, and part-aware without 3-grams | whether the tokenizer earns its index-size cost |
| De-duplication by confidence vs by area | **the most interesting one.** It should move evidence sufficiency while leaving ranking metrics roughly unchanged. If it moves ranking instead, or moves nothing, the rule isn't earning its place. |

A comparison against page-image (pixel) RAG is deliberately **not** on this list. It
would require matching a generation model and a visual encoder across systems, which
introduces more confounds than the comparison resolves.

---

## What would limit any result

- **One person.** Design, implementation, and any query authoring and annotation
  would share an author. Expectancy bias is not controlled for.
- **One document family.** Both corpora are crystal/oscillator datasheets from
  adjacent vendors.
- **Born-digital only.** No OCR, so scanned PDFs can't be in the query set.
- **Frozen enrichment.** Descriptions and TOC paths came from one VLM run;
  regenerating them changes the index, so a result measures one specific index.
- **Small corpora.** At 875 and 547 chunks, confidence intervals on Recall@1 are
  wide enough that bare point estimates would mislead.
