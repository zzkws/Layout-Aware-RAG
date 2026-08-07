# TKD Evidence RAG demo

Public static deployment of the original TKD demo structure. The corpus overview, search view, chunk browser, document tree, and evidence-dialog page read versioned static snapshots and require no hosted model.

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
