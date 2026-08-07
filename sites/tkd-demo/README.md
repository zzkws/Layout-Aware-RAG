# TKD Evidence RAG demo

Public, model-free technical exhibit for the TKD corpus. It runs deterministic BM25 in the browser and exposes the page-native evidence contract.

```bash
npm install
npm run dev
npm run build
npm test
```

Generate the compact public bundle from the repository root:

```bash
python -m tools.export_site_data --corpus tkd --data-dir data_tkd --out sites/tkd-demo/public/data
```
