# Information Retrieval Systems (IRS) 🔎

> ⚠️ **Project status:** preliminary version under active development. Structure, features, and results may evolve.

## 🧠 Overview

This repository implements a prototype of an **Information Retrieval System (IRS)** aimed at:

- 📥 Ingesting and organizing web documents.
- 🧹 Preprocessing content for indexing.
- 🗂️ Building specialized indexes (inverted trie + co-occurrence).
- 📊 Document ranking with the **Generalized Vector Space Model (GVSM)**.
- 🌐 Exploring results through a web interface.

## 🏗️ System Architecture

The system is divided into three main layers:

1. **Data pipeline**
   - Collection of raw resources (webpages, PDFs, images).
   - Extraction and normalization of content.

2. **Retrieval engine**
   - Building on-disk indexes from the processed corpus.
   - Computing semantic relationships and similarity scores with GVSM.

3. **Presentation layer**
   - Frontend application for search, filters, and result visualization.

## 📁 Repository Structure

```text
data/
  raw/                # Original resources (webpages, pdfs, images)
  extracted/          # Extracted and structured content
  processed/          # Indexing artifacts (JSON indexes)
scripts/
  indexer.py          # Index construction
  gvsm_model.py       # GVSM-based ranking model
webapp/
  src/                # React components and styles
  public/             # Static assets
```

## ⚙️ Key Components

- `scripts/indexer.py`
  - Generates index structures for efficient access to terms and documents.

- `scripts/gvsm_model.py`
  - Implements the generalized vector model for similarity-based ranking.

- `webapp/`
  - Frontend with **Vite + React** for query interaction and visual analysis.

## 🔄 Data Workflow

1. `data/raw` → input of original resources.
2. `data/extracted` → output from extraction/parsing.
3. `scripts/indexer.py` → generation of indexes in `data/processed`.
4. `scripts/gvsm_model.py` → computation of relevance for queries.
5. `webapp/` → interactive querying and result visualization.

## 🚀 Status and Goal

Project focused on academic experimentation and iterative improvement of IR techniques:

- ✅ Functional baseline for indexing and ranking.
- 🧪 Open space for experiments on retrieval quality.
- 📈 Planned improvements in performance, relevance, and search UX.

## 🔬 Search model (GVSM)

- **Summary:** The search engine uses the Generalized Vector Space Model (GVSM) to rank results. GVSM extends the classic vector space model by incorporating term–term correlations via a co-occurrence index.
- **Representation:** Queries and documents are represented as sparse vectors with weights $w_{t} = tf_{norm} \cdot idf(t)$, with

  $idf(t) = \log\left(\frac{N+1}{df(t)+1}\right) + 1$

  where $N$ is the total number of documents and $df(t)$ is the document frequency of term $t$.
### **Scoring (formula)**

$$
\mathrm{sim}(q,d) =
\frac{
\sum_{i\in q}\sum_{j\in d} w_{i,q}\, w_{j,d}\, s_{ij}
}{
\sqrt{\sum_{i\in q} w_{i,q}^2}
\;\;
\sqrt{\sum_{j\in d} w_{j,d}^2}
}
$$

### **Term–term correlation**

$s_{ij}$ is obtained from the co-occurrence index; by default a cosine-like normalization is used:

$$
s_{ij} = \frac{\mathrm{cooc}(i,j)}{\sqrt{df(i)\, df(j)}}
$$


- **Implementation:** See `scripts/gvsm_model.py` (classes `GeneralizedVectorSpaceModel` and `CoOccurrenceIndex`) for IDF, vector construction and the `similarity()` implementation.

## 🗂️ Indexes and data structures

- **Inverted trie (Patricia Trie):** persisted at [data/processed/inverted_index_trie.json](data/processed/inverted_index_trie.json). Main format:
  - Top-level: `{"nodes": [...], "root": 0, "count": <unique_tokens>, "doc_count": <N>}`.
  - Each node: `{"is_end": bool, "docs": [doc_id, ...], "children": {"edge_label": node_index, ...}}`.
  - `docs` stores the document IDs where the term appears (presence only). Per-document term frequencies are not embedded in each posting; frequencies are reconstructed from tokenized documents when building vectors.
  - The trie supports prefix search and candidate retrieval via `get_parcial_AND` / `intersect_tokens`.

- **Co-occurrence index:** persisted at [data/processed/cooccurrence_index.json](data/processed/cooccurrence_index.json). It contains fields `df`, `total_docs`, `cooc` (nested mapping term→term→count) and `min_cooc`.

## 🔧 Retrieval pipeline

1. Normalization and tokenization (spaCy) via `Index.tokenize()` in `scripts/indexer.py`.
2. Candidate retrieval with the `PatriciaTrie` (exact search or `get_parcial_AND` for multi-term queries).
3. Query vector construction: normalized TF × IDF (implemented in `get_query_vector`).
4. Document vectors precomputed (generated during `GVSMSearchEngine` initialization).
5. Re-ranking using GVSM and the $s_{ij}$ correlations from the co-occurrence index.
6. Return top-K results ordered by score.

## 🧾 Format, performance and recommendations

- Indexes are stored as JSON in `data/processed` (human-readable and portable). For larger collections we recommend migrating to binary formats (msgpack, sqlite, LMDB) or using `mmap` to reduce memory and speed up I/O.
- The system precomputes document vectors and caches the co-occurrence index in `cooccurrence_index.json` for fast queries.
- Future best practices: apply dimensionality reduction (SVD/LSA) over the co-occurrence matrix, vector quantization, or use inverted indexes with compressed postings for scalability.

## 🛠️ Local testing

- Rebuild/update indexes: `python scripts/indexer.py`
- Run integrated GVSM test: `python scripts/gvsm_model.py`

