
# Information Retrieval Systems (IRS) 🔎

> ⚠️ Project status: preliminary — active development and experimental.

## Overview 🧠

This repository implements a small experimental Information Retrieval System (IRS) focused on:

- 🕷️ Ingesting and cleaning web documents (focused crawler).
- 🗂️ Building lightweight indexes (Patricia trie inverted index + co-occurrence index).
- 📊 Ranking documents using the Generalized Vector Space Model (GVSM).
- 🌐 Exploring results via a React + Vite frontend.

The codebase is intended for research and experiments rather than production use.

## System architecture 🏗️

The project has three logical layers:

1. Data pipeline: crawling and extraction of web documents into JSONL.
2. Retrieval engine: index construction (trie + co-occurrence) and GVSM ranking.
3. Presentation layer: Vite + React webapp for interactive search.

## Repository layout 📁

```text
data/                 # raw/extracted resources and processed indexes
scripts/              # crawler, indexer and engine (Python)
webapp/               # React frontend (Vite)
requirements.txt      # Python dependencies
README.md             # this file
```

Key script entrypoints 🛠️:

- 🕸️ `scripts/tech_crawler.py` — focused tech crawler (EN+ES seeds) that writes `data/extracted/webpages/webpages.jsonl`.
- 🧾 `scripts/indexer.py` — builds/updates the PatriciaTrie inverted index (JSON persisted).
- 🔎 `scripts/main.py` — GVSM search pipeline and small Flask-based `/search` API (run with `serve`).

## Data artifacts 🗂️

- 📄 `data/extracted/webpages/webpages.jsonl` — newline-delimited JSON documents (fields include `doc_id` and `text`).
- 💾 `data/raw/webpages/` — optional saved raw HTML pages (when crawler run with `--save-raw`).
- 🧭 `data/processed/inverted_index_trie.json` — persisted Patricia Trie index.
- 🔗 `data/processed/cooccurrence_index.json` — persisted co-occurrence index (df, cooc counts, total_docs).

## Quick start (Windows) ⚙️

These steps assume a fresh clone on Windows. Use `py -3` when the `python` alias is not available.

1) Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
```

2) Upgrade pip and install Python dependencies:

```powershell
py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements.txt
```

3) Install the Spanish spaCy model used by the tokenizer (required):

```powershell
py -3 -m spacy download es_core_news_sm
```

4) (Optional) Install Node and dependencies to run the frontend:

Download and install Node.js (LTS recommended). Then:

```powershell
cd webapp
npm install
```

## Running the data pipeline and search 🚀

1) Crawl a technology-focused seed set (example) 🕷️:

```powershell
py -3 scripts/tech_crawler.py --max-pages 500 --max-depth 2 --save-raw --seeds-file scripts/tech_seeds.txt
```

This writes cleaned documents to `data/extracted/webpages/webpages.jsonl`. The crawler accepts English + Spanish pages together (no language filtering) and stores detected language only as metadata in `language`.

The crawler supports several useful flags:

- `--seeds-file` (default: `scripts/tech_seeds.txt`)
- `--output` (default: `data/extracted/webpages/webpages.jsonl`)
- `--raw-dir` (default: `data/raw/webpages`)
- `--max-pages`, `--max-depth`, `--min-chars`, `--per-domain-limit`, `--delay`
- `--only-new` (skip URLs/text already present in output)
- `--doc-id-mode` (`int` for indexer compatibility, `hash` optional)

2) Build or update the inverted trie index 🧾:

```powershell
py -3 scripts/indexer.py
```

This command will load the JSONL dataset (if present) and persist `data/processed/inverted_index_trie.json`.

3) Start the lightweight GVSM API (Flask) 🧩:

```powershell
py -3 scripts/main.py serve
```

Notes:
- 🔎 The HTTP API exposes `/search?query=...&top_k=...` and returns JSON results.
- 🗄️ On first run the co-occurrence index (`data/processed/cooccurrence_index.json`) is built automatically and cached. To force a rebuild, delete the cache file and restart the engine.
- ⚠️ If `Flask` is missing: `py -3 -m pip install flask`.

4) Start the frontend (development) 🌐:

```powershell
cd webapp
npm run dev
```

Open the Vite dev server (usually http://localhost:5173) and use the search UI. For a production build:

```powershell
cd webapp
npm run build
npm run preview
```

## How the retrieval pipeline works (brief) 🔬

1. 📝 Documents are tokenized with the Spanish `spaCy` model (`Index.tokenize()` in `scripts/indexer.py`).
2. 🔎 Terms are stored in a compressed Patricia Trie (inverted index) for fast candidate retrieval.
3. 🔗 A co-occurrence index (term × term) is computed and cached; document frequencies (`df`) are used to compute IDF.
4. 📐 Queries and documents are vectorized (normalized TF × IDF). GVSM scoring uses term–term correlations from the co-occurrence index to compute similarity.

## Troubleshooting & tips 💡

- 🪟 Use `py -3` on Windows if `python` is not available.
- ✔️ Ensure `es_core_news_sm` is installed for spaCy tokenization.
- ⚠️ If the API prints an error about Flask, install Flask as shown above.
- 🔁 If you want to rebuild indexes from scratch, remove `data/processed/inverted_index_trie.json` and `data/processed/cooccurrence_index.json` and re-run the indexer and engine.
- 📦 Node >= 16 / npm required for the frontend (Vite).

## Development notes 🛠️

- 🗣️ The tokenizer and stop-word list are tuned for Spanish content. Adjust `scripts/indexer.py` if indexing other languages.
- 📚 Indexes are persisted as JSON for portability. For larger corpora consider migrating to a binary format (sqlite, LMDB, msgpack) for performance.

---

See [scripts/indexer.py](scripts/indexer.py) and [scripts/main.py](scripts/main.py) for implementation details.

