(function () {
  "use strict";

  const nativeFetch = window.fetch.bind(window);
  let snapshotPromise;

  function loadSnapshot() {
    if (!snapshotPromise) {
      snapshotPromise = Promise.all([
        nativeFetch("/data/corpus.json").then((response) => response.json()),
        nativeFetch("/data/chunks.json").then((response) => response.json()),
        nativeFetch("/data/bm25.json").then((response) => response.json()),
        nativeFetch("/data/examples.json").then((response) => response.json()),
      ]).then(([corpus, chunks, bm25, examples]) => ({ corpus, chunks, bm25, examples }));
    }
    return snapshotPromise;
  }

  function tokenize(text) {
    const output = [];
    for (const rawValue of String(text || "").toLowerCase().split(/\s+/)) {
      const raw = rawValue.replace(/^[.,;:()[\]{}"']+|[.,;:()[\]{}"']+$/g, "");
      if (!raw) continue;
      const words = raw.match(/[a-z0-9]+/g) || [];
      const looksLikePart = /^(?=.*\d)[a-z0-9][a-z0-9.\-_/卤]{4,}$/.test(raw);
      if (looksLikePart) {
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

  function search(query, index, limit) {
    const scores = new Float64Array(index.n_docs);
    for (const token of new Set(tokenize(query))) {
      const postings = index.postings[token];
      if (!postings || !postings.length) continue;
      const idf = Math.log(1 + (index.n_docs - postings.length + 0.5) / (postings.length + 0.5));
      for (const [documentIndex, frequency] of postings) {
        const length = index.doc_lens[documentIndex] || index.avg_len;
        scores[documentIndex] += idf *
          ((frequency * 2.5) / (frequency + 1.5 * (0.25 + (0.75 * length) / index.avg_len)));
      }
    }
    return Array.from(scores, (score, indexValue) => ({ index: indexValue, score }))
      .filter((item) => item.score > 0)
      .sort((left, right) => right.score - left.score)
      .slice(0, limit || 15);
  }

  function compatChunk(chunk) {
    return {
      chunk_id: chunk.id,
      doc_id: chunk.doc_id,
      page: chunk.page,
      block_type: chunk.type,
      section_title: chunk.title,
      toc_path: chunk.toc,
      description: chunk.description,
      keywords: chunk.keywords || [],
      native_text: chunk.text || "",
      n_elements: (chunk.bboxes || []).length,
      bboxes_pdf: chunk.bboxes || [],
      crop_images: [],
      merged_image: chunk.image_available
        ? `../${String(chunk.evidence_image).replace(/^\//, "")}`
        : null,
      source_pdf: chunk.source_pdf,
    };
  }

  function rankedChunks(query, snapshot, limit) {
    return search(query, snapshot.bm25, limit).map((item, rank) => ({
      ...compatChunk(snapshot.chunks[item.index]),
      rrf_score: item.score,
      dense_rank: "offline",
      bm25_rank: rank + 1,
      bm25_score: item.score,
    }));
  }

  function stats(snapshot) {
    return {
      total_docs: snapshot.corpus.documents,
      total_pages: snapshot.corpus.pages,
      total_chunks: snapshot.corpus.chunks,
      embed_model: "public/static-snapshot",
      vlm_model: "offline ingestion",
      block_types: snapshot.corpus.block_types,
      docs: snapshot.corpus.document_rows.map((row) => ({
        doc_id: row.id,
        title: row.title,
        series: row.series,
        product_type: row.product_type,
        frequency_range: row.frequency_range,
        package: row.package,
        pages: row.pages,
        chunks: row.chunks,
      })),
    };
  }

  function tableOfContents(chunks) {
    const output = [];
    const seen = new Set();
    for (const chunk of chunks) {
      const parts = String(chunk.toc || chunk.title || "Untitled section")
        .split(" > ")
        .filter(Boolean);
      let path = "";
      parts.forEach((title, index) => {
        path = path ? `${path} > ${title}` : title;
        if (!seen.has(path)) {
          seen.add(path);
          output.push({ level: index + 1, title, page: chunk.page });
        }
      });
    }
    return output;
  }

  function documentPayload(docId, snapshot) {
    const source = snapshot.chunks.filter((chunk) => chunk.doc_id === docId);
    if (!source.length) return { error: `unknown doc: ${docId}` };
    const row = snapshot.corpus.document_rows.find((item) => item.id === docId) || {};
    return {
      doc_id: docId,
      source_pdf: source[0].source_pdf,
      doc_card: {
        doc_name: row.title || docId,
        product_type: row.product_type || "",
        frequency_range: row.frequency_range || "",
        package_size_mm: row.package || "",
        summary: `${row.series || docId} · static public corpus snapshot`,
        toc: tableOfContents(source),
      },
      chunks: source.map(compatChunk),
    };
  }

  function evidenceSummary(request, results, manufacturer) {
    if (!results.length) return `静态快照中没有找到与“${request}”匹配的证据。`;
    const lines = results.slice(0, 3).map((result, index) =>
      `- [E${index + 1}] ${result.doc_id}.pdf 第 ${result.page} 页：${result.description || result.native_text.slice(0, 180)}`
    );
    return `这是 ${manufacturer} 公开静态快照中的证据摘要，不是在线模型生成结果。\n\n${lines.join("\n")}`;
  }

  function jsonResponse(payload, status) {
    return new Response(JSON.stringify(payload), {
      status: status || 200,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }

  window.fetch = async function (input, init) {
    const requestUrl = new URL(typeof input === "string" ? input : input.url, window.location.href);
    if (!requestUrl.pathname.startsWith("/api/")) return nativeFetch(input, init);

    const snapshot = await loadSnapshot();
    const path = requestUrl.pathname;
    let body = {};
    if (init && init.body) {
      try { body = JSON.parse(init.body); } catch { body = {}; }
    }

    if (path === "/api/stats") return jsonResponse(stats(snapshot));
    if (path === "/api/chunks") {
      return jsonResponse({ chunks: snapshot.chunks.map(compatChunk) });
    }
    if (path === "/api/doc") {
      const payload = documentPayload(requestUrl.searchParams.get("id") || "", snapshot);
      return jsonResponse(payload, payload.error ? 404 : 200);
    }
    if (path === "/api/search" || path === "/api/ask") {
      const query = String(body.query || body.request || "").trim();
      const results = rankedChunks(query, snapshot, Number(body.top_k) || 15);
      return jsonResponse({
        request: body.request || query,
        query,
        note: "公开站读取随版本发布的静态快照；不调用查询改写或云端生成模型。",
        results,
      });
    }
    if (path === "/api/answer") {
      const byId = new Map(snapshot.chunks.map((chunk) => [chunk.id, chunk]));
      const results = (body.chunk_ids || []).map((id) => byId.get(id)).filter(Boolean).map(compatChunk);
      return jsonResponse({
        answer: evidenceSummary(body.request || body.query || "当前问题", results, snapshot.corpus.manufacturer),
        model: "static evidence snapshot",
      });
    }
    if (path === "/api/chat") {
      const messages = body.messages || [];
      const query = String(messages[messages.length - 1]?.content || "").trim();
      const results = rankedChunks(query, snapshot, 3);
      const text = evidenceSummary(query, results, snapshot.corpus.manufacturer);
      const event = `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\ndata: [DONE]\n\n`;
      return new Response(event, { headers: { "Content-Type": "text/event-stream; charset=utf-8" } });
    }
    return jsonResponse({ error: "not found" }, 404);
  };
})();
