"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { siteConfig } from "./site-config";

type DocumentRow = {
  id: string;
  title: string;
  series: string;
  product_type: string;
  frequency_range: string;
  package: string;
  chunks: number;
  pages: number;
};

type Corpus = {
  title: string;
  manufacturer: string;
  documents: number;
  chunks: number;
  pages: number;
  block_types: [string, number][];
  document_rows: DocumentRow[];
  public_mode: string;
};

type Chunk = {
  id: string;
  doc_id: string;
  page: number;
  type: string;
  title: string;
  toc: string;
  description: string;
  text: string;
  keywords: string[];
  bboxes: number[][];
  source_pdf: string;
  evidence_image: string;
  image_available: boolean;
};

type Bm25 = {
  n_docs: number;
  avg_len: number;
  doc_lens: number[];
  postings: Record<string, [number, number][]>;
};

type SearchResult = { index: number; score: number };

const partLike = /^(?=.*\d)[a-z0-9][a-z0-9.\-_/±]{4,}$/;

function tokenize(text: string) {
  const output: string[] = [];
  for (const rawValue of text.toLowerCase().split(/\s+/)) {
    const raw = rawValue.replace(/^[.,;:()[\]{}"']+|[.,;:()[\]{}"']+$/g, "");
    if (!raw) continue;
    const words = raw.match(/[a-z0-9]+/g) ?? [];
    if (partLike.test(raw)) {
      output.push(raw, ...words);
      const compact = raw.replace(/[-_]/g, "");
      for (let index = 0; index <= compact.length - 3; index += 1) {
        output.push(compact.slice(index, index + 3));
      }
    } else {
      output.push(...words);
    }
  }
  return output;
}

function bm25Search(query: string, index: Bm25, limit = 8): SearchResult[] {
  const scores = new Float64Array(index.n_docs);
  const tokens = new Set(tokenize(query));
  for (const token of tokens) {
    const postings = index.postings[token];
    if (!postings?.length) continue;
    const idf = Math.log(1 + (index.n_docs - postings.length + 0.5) / (postings.length + 0.5));
    for (const [documentIndex, frequency] of postings) {
      const length = index.doc_lens[documentIndex] || index.avg_len;
      const normalized =
        (frequency * 2.5) / (frequency + 1.5 * (0.25 + (0.75 * length) / index.avg_len));
      scores[documentIndex] += idf * normalized;
    }
  }
  return Array.from(scores, (score, arrayIndex) => ({ index: arrayIndex, score }))
    .filter((result) => result.score > 0)
    .sort((left, right) => right.score - left.score)
    .slice(0, limit);
}

export default function Home() {
  const [corpus, setCorpus] = useState<Corpus | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [bm25, setBm25] = useState<Bm25 | null>(null);
  const [query, setQuery] = useState(siteConfig.exampleQuery);
  const [submittedQuery, setSubmittedQuery] = useState(siteConfig.exampleQuery);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/data/corpus.json").then((response) => response.json()),
      fetch("/data/chunks.json").then((response) => response.json()),
      fetch("/data/bm25.json").then((response) => response.json()),
    ])
      .then(([corpusPayload, chunkPayload, bm25Payload]) => {
        setCorpus(corpusPayload);
        setChunks(chunkPayload);
        setBm25(bm25Payload);
        setResults(bm25Search(siteConfig.exampleQuery, bm25Payload));
      })
      .finally(() => setLoading(false));
  }, []);

  const maxTypeCount = useMemo(
    () => Math.max(...(corpus?.block_types.map(([, count]) => count) ?? [1])),
    [corpus],
  );

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const clean = query.trim();
    setSubmittedQuery(clean);
    setResults(clean && bm25 ? bm25Search(clean, bm25) : []);
  }

  return (
    <main>
      <header className="nav-shell">
        <a className="brand" href="#top" aria-label="Evidence RAG home">
          <span className="brand-mark" aria-hidden="true">E</span>
          <span>Evidence RAG Pilot</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#demonstrates">Principles</a>
          <a href="#search">Search</a>
          <a href="#architecture">Architecture</a>
          <a href="#requirements">Requirements</a>
          <a href={siteConfig.githubUrl} target="_blank" rel="noreferrer">GitHub</a>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">{siteConfig.corpus} technical prototype · public demo</p>
          <h1>Datasheet answers should point back to the page.</h1>
          <p className="hero-lede">
            A portfolio demonstration of page-native RAG for engineering documents.
            Every chunk keeps its section, page, bounding boxes, text, and visual evidence together.
          </p>
          <div className="hero-actions">
            <a className="button primary" href="#search">Explore the corpus</a>
            <a className="button secondary" href={siteConfig.githubUrl} target="_blank" rel="noreferrer">
              Read the implementation
            </a>
          </div>
        </div>
        <div className="evidence-card" aria-label="Evidence chunk anatomy">
          <div className="card-topline"><span>TRACEABLE CHUNK</span><span>p. 04</span></div>
          <div className="paper-block title-line" />
          <div className="paper-grid">
            <div className="paper-block table-block" />
            <div className="paper-block diagram-block" />
          </div>
          <div className="bbox-label">bbox_pdf · section · source</div>
          <div className="trace-line"><span>E1</span><i /><b>answer claim</b></div>
        </div>
      </section>

      <section className="metrics" aria-label="Corpus statistics">
        <div><strong>{corpus?.documents ?? "—"}</strong><span>indexed documents</span></div>
        <div><strong>{corpus?.pages ?? "—"}</strong><span>source pages</span></div>
        <div><strong>{corpus?.chunks ?? "—"}</strong><span>evidence chunks</span></div>
        <div><strong>0</strong><span>visitor API keys</span></div>
      </section>

      <section className="principles" id="demonstrates">
        <div className="section-heading">
          <p className="eyebrow">What this project demonstrates</p>
          <h2>The engineering ideas are the exhibit.</h2>
          <p>
            The goal is to make the design inspectable—from page geometry to a cited evidence package—not to present an opaque chatbot.
          </p>
        </div>
        <div className="principle-grid">
          <article><span>01</span><h3>Page-native chunks</h3><p>Semantic regions keep their original coordinates, reading context, and visual form.</p></article>
          <article><span>02</span><h3>Auditable retrieval</h3><p>Search results expose ranking, source page, bbox count, text, and evidence image.</p></article>
          <article><span>03</span><h3>Replaceable models</h3><p>The evidence contract stays stable while layout, embedding, and answer models evolve.</p></article>
        </div>
        <p className="scope-note"><strong>Scope:</strong> a technical prototype and portfolio project—not a SOTA claim or a benchmark comparison with Pixel RAG or other systems.</p>
      </section>

      <section className="search-section" id="search">
        <div className="section-heading">
          <p className="eyebrow">Live deterministic retrieval</p>
          <h2>Search the {siteConfig.corpus} evidence index</h2>
          <p>
            This public demo runs BM25 in your browser. The full repository adds dense retrieval and RRF;
            no visitor query is sent to a model provider.
          </p>
        </div>
        <form className="search-form" onSubmit={submitSearch}>
          <label htmlFor="query">Part number, parameter, package, or engineering constraint</label>
          <div className="search-row">
            <input
              id="query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="e.g. load capacitance 12pF ESR"
              autoComplete="off"
            />
            <button type="submit">Search evidence</button>
          </div>
        </form>

        <div className="result-summary" aria-live="polite">
          {loading ? "Loading local index…" : submittedQuery ? `${results.length} results for “${submittedQuery}”` : "Enter a query"}
        </div>
        <div className="results-grid">
          {results.map((result, rank) => {
            const chunk = chunks[result.index];
            if (!chunk) return null;
            return (
              <article className="result-card" key={chunk.id}>
                <div className="result-rank">E{rank + 1}</div>
                <div className="result-body">
                  <div className="result-meta">
                    <span>{chunk.doc_id}.pdf · p.{chunk.page}</span>
                    <span>{chunk.type.replaceAll("_", " ")}</span>
                    <span>BM25 {result.score.toFixed(2)}</span>
                  </div>
                  <h3>{chunk.title || chunk.toc || chunk.id}</h3>
                  <p>{chunk.description || chunk.text.slice(0, 260)}</p>
                  <details>
                    <summary>Inspect source evidence</summary>
                    {chunk.image_available ? (
                      // Evidence crops have source-dependent dimensions; preserve them without optimization.
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={chunk.evidence_image} alt={`Evidence crop for ${chunk.id}`} loading="lazy" />
                    ) : (
                      <p className="asset-note">The full-resolution crop is included in the GitHub data release.</p>
                    )}
                    <dl>
                      <div><dt>Chunk</dt><dd>{chunk.id}</dd></div>
                      <div><dt>TOC path</dt><dd>{chunk.toc || "—"}</dd></div>
                      <div><dt>Bounding boxes</dt><dd>{chunk.bboxes.length}</dd></div>
                    </dl>
                  </details>
                </div>
              </article>
            );
          })}
        </div>
      </section>

      <section className="architecture" id="architecture">
        <div className="section-heading inverse">
          <p className="eyebrow">The core design</p>
          <h2>From PDF geometry to claim-ready evidence</h2>
          <p>Token windows disappear. Page regions remain inspectable throughout retrieval.</p>
        </div>
        <ol className="pipeline-list">
          {[
            ["01", "Render", "Preserve page scale and PDF coordinates."],
            ["02", "Detect", "Find tables, figures, captions, and text blocks."],
            ["03", "Group", "Build semantic page-native chunks with code fallbacks."],
            ["04", "Index", "Combine dense retrieval, model-aware BM25, and RRF."],
            ["05", "Cite", "Return page, bbox, text, and merged evidence image."],
          ].map(([number, title, description]) => (
            <li key={number}><span>{number}</span><h3>{title}</h3><p>{description}</p></li>
          ))}
        </ol>
      </section>

      <section className="corpus-section">
        <div className="section-heading">
          <p className="eyebrow">Corpus anatomy</p>
          <h2>What the index contains</h2>
        </div>
        <div className="corpus-layout">
          <div className="type-chart" aria-label="Chunk types">
            {(corpus?.block_types.slice(0, 8) ?? []).map(([type, count]) => (
              <div className="type-row" key={type}>
                <span>{type.replaceAll("_", " ")}</span>
                <i><b style={{ width: `${(count / maxTypeCount) * 100}%` }} /></i>
                <em>{count}</em>
              </div>
            ))}
          </div>
          <div className="document-list">
            {(corpus?.document_rows.slice(0, 8) ?? []).map((document) => (
              <article key={document.id}>
                <strong>{document.series || document.id}</strong>
                <span>{document.product_type || "datasheet"}</span>
                <small>{document.pages} pages · {document.chunks} chunks</small>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="requirements" id="requirements">
        <div className="section-heading">
          <p className="eyebrow">Deployment reality</p>
          <h2>Three operating profiles, one evidence contract</h2>
          <p>The public explorer proves the information design without pretending a browser is a 12B inference server.</p>
        </div>
        <div className="profile-grid">
          <article className="featured"><span>PUBLIC SITES</span><h3>No GPU · no API key</h3><p>Live BM25, corpus browsing, and static evidence snapshots. Best for reliable public sharing.</p></article>
          <article><span>LOCAL LITE</span><h3>CPU · 8–16GB RAM</h3><p>BGE-small or Harrier for live retrieval. Answer generation stays optional.</p></article>
          <article><span>FULL PIPELINE</span><h3>32GB min · 48GB preferred</h3><p>Layout detection, 0.6B embedding, offline VLM enrichment, and a 12B multimodal answer model.</p></article>
        </div>
        <a className="report-link" href={siteConfig.reportUrl} target="_blank" rel="noreferrer">
          Read the bilingual technical report <span>↗</span>
        </a>
      </section>

      <footer>
        <div><strong>Evidence RAG Pilot</strong><p>Original engineering-document RAG design by zzkws.</p></div>
        <div><span>{siteConfig.corpus} snapshot</span><span>Apache-2.0 code</span><a href={siteConfig.githubUrl}>GitHub repository</a></div>
      </footer>
    </main>
  );
}
