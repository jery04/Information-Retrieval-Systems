# Information Retrieval Systems (IRS)

> **Preliminary Version:** This project is currently under active development, and features, structure, and behavior may change.

## About the Project

This repository contains an Information Retrieval System prototype focused on indexing and ranking web content.

The project combines:
- Document ingestion and preprocessing pipelines.
- Index construction (including trie-based and co-occurrence-based structures).
- A Generalized Vector Space Model (GVSM) for document ranking.
- A web interface for searching and exploring results.

## Project Structure

- `data/`
  - `raw/`: Original collected resources (web pages, PDFs, images).
  - `extracted/`: Parsed or extracted content ready for indexing.
  - `processed/`: Generated artifacts such as index files.
- `scripts/`
  - Core retrieval logic and indexing/ranking scripts.
  - Includes modules such as `indexer.py` and `gvsm_model.py`.
- `webapp/`
  - Frontend application (Vite + React) for the search interface.
  - Contains UI components, styles, and static assets.

## Current Status

This is an early-stage implementation intended for experimentation and iterative improvement in Information Retrieval workflows.
