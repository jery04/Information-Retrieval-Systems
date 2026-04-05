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
