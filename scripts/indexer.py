"""Lightweight indexing utilities: Spanish tokenizer and Trie inverted index.

This module provides simple helpers for Spanish text preprocessing and a
Trie-based inverted index with JSON persistence. Docstrings are brief and
focused on functionality (main() is intentionally unchanged).
"""

import unicodedata
import re
import json
import time
import os
from typing import Any, Dict, List, Optional, Set
import spacy

# Spanish stop words set.
stop_words: Set[str] = {
    'a', 'acaso', 'ademas', 'adonde', 'ahi', 'ahora', 'al', 'algo', 'algun',
    'alguna', 'algunas', 'alguno', 'algunos', 'alla', 'alli', 'ambos', 'ante',
    'antes', 'apenas', 'aquel', 'aquella', 'aquellas', 'aquello', 'aquellos',
    'aqui', 'asi', 'aun', 'aunque',

    'bajo', 'bastante',

    'cabe', 'cada', 'casi', 'cierto', 'como', 'con', 'contra', 'cual',
    'cuales', 'cualquier', 'cualquiera', 'cuando', 'cuanta', 'cuantas',
    'cuanto', 'cuantos',

    'dar', 'de', 'deber', 'deberia', 'debo', 'debes', 'debe', 'debemos',
    'deben', 'decir', 'dejar', 'del', 'demas', 'demasiado', 'demasiada',
    'demasiados', 'demasiadas', 'desde', 'donde', 'dos', 'durante',

    'e', 'el', 'ella', 'ellas', 'ello', 'ellos', 'en', 'encima', 'encontrar',
    'entonces', 'entre', 'era', 'eran', 'eramos', 'eres', 'es', 'esa', 'esas',
    'ese', 'eso', 'esos', 'esta', 'estaba', 'estaban', 'estamos', 'estan',
    'estar', 'estas', 'este', 'esto', 'estos', 'estoy',

    'fin', 'fue', 'fueron',

    'haber', 'habia', 'habian', 'hace', 'hacen', 'hacer', 'hacia', 'hasta',
    'hay', 'he', 'hemos', 'han', 'has', 'ha', 'haya', 'hubo',

    'igual', 'incluso', 'ir', 'iba', 'iban',

    'jamas', 'junto',

    'la', 'las', 'le', 'les', 'llegar', 'llevar', 'lo', 'los', 'luego',
    'llamar',

    'mas', 'me', 'medio', 'mientras', 'mi', 'mia', 'mias', 'mio', 'mios',
    'mis', 'mucho', 'muchos', 'muy',

    'nada', 'nadie', 'ni', 'ningun', 'ninguna', 'ninguno', 'ningunos',
    'ningunas', 'no', 'nos', 'nosotras', 'nosotros', 'nuestra', 'nuestras',
    'nuestro', 'nuestros', 'nunca',

    'o', 'os', 'otra', 'otras', 'otro', 'otros',

    'para', 'parecer', 'pasar', 'pero', 'poco', 'poder', 'podemos', 'puedo',
    'puedes', 'puede', 'pueden', 'poner', 'por', 'porque', 'pronto', 'pues',

    'que', 'quedar', 'querer', 'quien', 'quienes', 'quizas', 'quiza',

    'saber', 'salvo', 'se', 'seguir', 'segun', 'ser', 'seran', 'sera',
    'si', 'siempre', 'sin', 'sino', 'sobre', 'soy', 'su', 'sus', 'suyo',
    'suya', 'suyos', 'suyas',

    'tal', 'tales', 'talvez', 'tambien', 'tanto', 'tan', 'te', 'temprano',
    'tener', 'tenemos', 'tengo', 'tienes', 'tiene', 'tienen', 'ti', 'tiempo',
    'todavia', 'todo', 'todos', 'tras', 'tu', 'tus', 'tuyo', 'tuya', 'tuyos',
    'tuyas',

    'un', 'una', 'unas', 'uno', 'unos', 'usted', 'ustedes',

    'va', 'vais', 'vamos', 'van', 'varios', 'ver', 'vez', 'vosotras',
    'vosotros', 'voy', 'vuestro', 'vuestra', 'vuestros', 'vuestras',

    'ya', 'yo'
}

class Index:
    """Spanish text preprocessing helper using spaCy.

    Provides tokenization, normalization and basic filtering for Spanish
    text. Tokens are lemmatized, lowercased, accent-stripped and filtered
    for punctuation, short tokens and stopwords.
    """
    nlp = spacy.load("es_core_news_sm")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize `text` and return a list of cleaned tokens.

        Steps: lemmatize, lowercase, strip accents, remove punctuation/space
        tokens, filter out short/non-alpha tokens and stopwords.
        """
        tokens: List[str] = []
        if not text:
            return tokens

        doc = Index.nlp(text)
        for token in doc:
            # skip punctuation and whitespace
            if token.is_punct or token.is_space:
                continue
            # lemmatize and lowercase
            word = token.lemma_.lower()
            # remove diacritics
            word = unicodedata.normalize("NFKD", word)
            word = "".join(ch for ch in word if not unicodedata.combining(ch))
            # filter short, non-alphabetic or stopword tokens
            if len(word) <= 1:
                continue
            if not word.isalpha():
                continue
            if token.is_stop or (word in stop_words):
                continue
            tokens.append(word)

        return tokens

class TrieNode:
    """Single node in a Trie storing child links and document ids."""
    def __init__(self):
        """Initialize node with empty children, end flag and doc list."""
        self.children = {}           # type: dict[str, TrieNode]
        self.is_end_of_word = False  # marks end of a word
        self.doc_ids = []            # list of document ids where the word appears

class Trie:
    """Trie-based inverted index with JSON (de)serialization utilities.

    Supports inserting words with optional document ids, exact-term search,
    and saving/loading the entire structure to/from a compact JSON format.
    """
    def __init__(self, filepath: Optional[str] = None):
        """Create a Trie and set default file path for persistence."""
        self.root = TrieNode()
        self.word_count = 0
        # default path if not provided
        if filepath is None:
            self.filepath = os.path.join("data", "processed", "inverted_index_trie.json")
        else:
            self.filepath = filepath

    def insert(self, word: str, doc_id: Optional[int] = None) -> None:
        """Insert `word` into the trie and optionally associate `doc_id`.

        Prevents duplicate doc ids for the same terminal node.
        """
        node = self.root
        # Traverse the trie, creating child nodes as needed for each character.
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]

        # If this node finishes a new word, mark it and increment the count.
        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1

        # Add the document id to the terminal node if provided and not already present.
        if doc_id is not None and doc_id not in node.doc_ids:
            node.doc_ids.append(doc_id)

    def search(self, word: str) -> Optional[TrieNode]:
        """Return the terminal node for exact `word`, or None if not found."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node if node.is_end_of_word else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Trie into a compact dict suitable for JSON storage.

        Uses a flat nodes list with index-based child references to keep JSON
        shallow and efficient to parse.
        """
        # flat serialization: list of nodes with index references
        nodes: List[Dict[str, Any]] = []
        node_index: Dict[int, int] = {}

        # BFS from the root; assign indices as nodes are discovered
        from collections import deque
        q = deque()
        q.append(self.root)
        node_index[id(self.root)] = 0
        nodes.append({"is_end": self.root.is_end_of_word, "docs": self.root.doc_ids[:], "children": {}})

        while q:
            node = q.popleft()
            idx = node_index[id(node)]
            for ch, child in node.children.items():
                cid = id(child)
                if cid not in node_index:
                    node_index[cid] = len(nodes)
                    nodes.append({"is_end": child.is_end_of_word, "docs": child.doc_ids[:], "children": {}})
                    q.append(child)
                nodes[idx]["children"][ch] = node_index[cid]

        # Include the word count for persistence
        return {"nodes": nodes, "root": 0, "count": self.word_count}

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Reconstruct the Trie from serialized `data` and apply to self.

        If `data` contains no nodes, the trie is reset to an empty state.
        """
        nodes_data = data.get("nodes", [])

        if not nodes_data:
            self.root = TrieNode()
            self.word_count = 0
            return

        nodes: List[TrieNode] = [TrieNode() for _ in nodes_data]
        for i, nd in enumerate(nodes_data):
            nodes[i].is_end_of_word = nd.get("is_end", False)
            nodes[i].doc_ids = nd.get("docs", []).copy()

        for i, nd in enumerate(nodes_data):
            for ch, child_idx in nd.get("children", {}).items():
                nodes[i].children[ch] = nodes[child_idx]

        # Assign reconstructed root and recalculate the number of indexed words.
        self.root = nodes[data.get("root", 0)]
        self.word_count = sum(1 for n in nodes if n.is_end_of_word)
        return

    def save(self, filepath: Optional[str] = None) -> None:
        """Save the Trie to a JSON file at `filepath` (or default path)."""
        if filepath is None:
            filepath = self.filepath

        # Ensure the directory exists
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            # Use compact separators to save space and speed I/O
            json.dump(self.to_dict(), f, ensure_ascii=False, separators=(",", ":"))

    def load(self, filepath: Optional[str] = None) -> None:
        """Load the Trie from a JSON file and apply it to this instance.

        If the target JSON file does not exist, build the trie from the
        original JSONL data source at data/extracted/webpages/sample_webpages.jsonl
        by tokenizing each line's `text` and inserting tokens with `doc_id`.
        The constructed trie is saved to `filepath`.
        """
        if filepath is None:
            filepath = self.filepath

        # If the specified file exists, load normally.
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.from_dict(data)
            return

        # Fallback: build trie from original JSONL dataset.
        sample_path = os.path.join("data", "extracted", "webpages", "sample_webpages.jsonl")
        if not os.path.exists(sample_path):
            # No data to build from; reset to empty trie.
            self.root = TrieNode()
            self.word_count = 0
            return

        # Reset trie before building
        self.root = TrieNode()
        self.word_count = 0

        with open(sample_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = int(obj.get("doc_id"))
                text = obj.get("text", "")
                # try to convert numeric doc_id strings to int
                try:
                    doc_id = int(doc_id)
                except Exception:
                    pass
                tokens = Index.tokenize(text)
                for token in set(tokens):
                    self.insert(token, doc_id)

        # Persist the newly created trie to the requested filepath.
        self.save(filepath)

def main():
    trie = Trie()
    trie.load()  # will build from JSONL if the trie JSON doesn't exist
    print(f"Indexed {trie.word_count} unique tokens in the Trie inverted index.")


if __name__ == "__main__":
    main()