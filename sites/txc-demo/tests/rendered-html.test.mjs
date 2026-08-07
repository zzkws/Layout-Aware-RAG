import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function loadWorker() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  return (await import(workerUrl.href)).default;
}

const assets = {
  async fetch(request) {
    const pathname = new URL(request.url).pathname.replace(/^\//, "");
    try {
      return new Response(await readFile(new URL(`../dist/client/${pathname}`, import.meta.url)), { status: 200 });
    } catch {
      return new Response("Not found", { status: 404 });
    }
  },
};

test("serves the original TXC demo structure from static assets", async () => {
  const worker = await loadWorker();
  for (const [route, marker] of [["/", "TXC Datasheet"], ["/chunks", "Chunk 浏览"], ["/doc", "目录树"], ["/chat", "静态证据"]]) {
    const response = await worker.fetch(new Request(`http://localhost${route}`), { ASSETS: assets }, {});
    assert.equal(response.status, 200, route);
    const html = await response.text();
    assert.match(html, new RegExp(marker));
    assert.match(html, /legacy-adapter\.js/);
  }
});

test("ships a static, keyless corpus adapter", async () => {
  const source = await readFile(new URL("../public/legacy-adapter.js", import.meta.url), "utf8");
  assert.match(source, /public\/static-snapshot/);
  assert.match(source, /\/data\/chunks\.json/);
  assert.doesNotMatch(source, /api[_-]?key|Authorization|Bearer/i);
});
