"""Lightweight indexing utilities: Spanish tokenizer and PatriciaTrie inverted index.

This module provides simple helpers for Spanish text preprocessing and a
PatriciaTrie-based inverted index with JSON persistence. Docstrings are brief and
focused on functionality (main() is intentionally unchanged).
"""

import unicodedata
import json
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
            if len(word) <= 2 or not word.isalpha():
                continue
            if token.is_stop or (word in stop_words):
                continue
            tokens.append(word)

        return tokens

class Node:
    """Single node in a Patricia Trie storing compressed edges."""
    def __init__(self):
        """Initialize node with empty children, end flag and doc list."""
        self.children = {}           # type: dict[str, Node]
        self.is_end_of_word = False  # marks end of a word
        self.doc_ids = []            # list of document ids where the word appears

class PatriciaTrie:
    """Patricia Trie inverted index with JSON (de)serialization utilities.

    Supports inserting words with optional document ids, exact-term search,
    and saving/loading the entire structure to/from a compact JSON format.
    """
    def __init__(self, filepath: Optional[str] = None):
        """Create a Trie and set default file path for persistence."""
        self.root = Node()
        self.word_count = 0
        # default path if not provided
        if filepath is None:
            self.filepath = os.path.join("data", "processed", "inverted_index_trie.json")
        else:
            self.filepath = filepath

    @staticmethod
    def _common_prefix_length(a: str, b: str) -> int:
        """Return the length of the common prefix shared by `a` and `b`."""
        i = 0
        max_i = min(len(a), len(b))
        while i < max_i and a[i] == b[i]:
            i += 1
        return i

    def insert(self, word: str, doc_id: Optional[int] = None) -> None:
        """Insert `word` into the Patricia Trie and optionally add `doc_id`.

        Child edges store substrings (not single chars), splitting edges when
        only a partial prefix matches.
        """
        if not word:
            return

        node = self.root
        remaining = word

        while remaining:
            matched = False
            for edge, child in list(node.children.items()):
                common_len = self._common_prefix_length(remaining, edge)
                if common_len == 0:
                    continue

                matched = True

                # Full edge match: descend and continue consuming the word.
                if common_len == len(edge):
                    node = child
                    remaining = remaining[common_len:]
                    break

                # Partial edge match: split the existing edge.
                prefix = edge[:common_len]
                old_suffix = edge[common_len:]
                new_suffix = remaining[common_len:]

                split_node = Node()
                split_node.children[old_suffix] = child

                del node.children[edge]
                node.children[prefix] = split_node

                # New word ends exactly at the split node.
                if not new_suffix:
                    if not split_node.is_end_of_word:
                        split_node.is_end_of_word = True
                        self.word_count += 1
                    if doc_id is not None and doc_id not in split_node.doc_ids:
                        split_node.doc_ids.append(doc_id)
                    return

                # Add a new branch for the remaining suffix.
                new_leaf = Node()
                new_leaf.is_end_of_word = True
                if doc_id is not None:
                    new_leaf.doc_ids.append(doc_id)
                split_node.children[new_suffix] = new_leaf
                self.word_count += 1
                return

            if not matched:
                # No shared prefix from this node: create a direct compressed edge.
                new_leaf = Node()
                new_leaf.is_end_of_word = True
                if doc_id is not None:
                    new_leaf.doc_ids.append(doc_id)
                node.children[remaining] = new_leaf
                self.word_count += 1
                return

        # Word already represented by the current node path.
        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1
        if doc_id is not None and doc_id not in node.doc_ids:
            node.doc_ids.append(doc_id)

    def search(self, word: str) -> Optional[Node]:
        """Return terminal node for exact `word`, or None if not found."""
        if not word:
            return None

        node = self.root
        remaining = word

        while remaining:
            matched = False
            for edge, child in node.children.items():
                if remaining.startswith(edge):
                    remaining = remaining[len(edge):]
                    node = child
                    matched = True
                    break

            if not matched:
                return None

        return node if node.is_end_of_word else None

    def get_all_words(self) -> List[str]:
        """Return a list with all words currently stored in the Trie."""
        words: List[str] = []

        def _collect(node: Node, prefix: str) -> None:
            if node.is_end_of_word:
                words.append(prefix)
            for edge, child in node.children.items():
                _collect(child, prefix + edge)

        _collect(self.root, "")
        words.sort()
        return words

    def print_tree(self, max_depth: Optional[int] = None, show_docs: bool = True) -> None:
        """Print the Trie structure showing compressed edges and node flags.

        Each line shows an edge label (the substring stored on the edge).
        Nodes that mark the end of a word are annotated with '*' and, when
        available, the `doc_ids` list is shown if `show_docs` is True.
        """
        def _print(node: Node, prefix: str, depth: int) -> None:
            if max_depth is not None and depth > max_depth:
                return
            children = list(node.children.items())
            for i, (edge, child) in enumerate(children):
                last = (i == len(children) - 1)
                connector = "└─" if last else "├─"
                line = f"{prefix}{connector}{edge}"
                if child.is_end_of_word:
                    line += " *"
                if show_docs and child.doc_ids:
                    line += " " + str(child.doc_ids)
                print(line)
                new_prefix = prefix + ("   " if last else "│  ")
                _print(child, new_prefix, depth + 1)

        root_info = "root"
        if self.root.is_end_of_word:
            root_info += " *"
        if show_docs and self.root.doc_ids:
            root_info += " " + str(self.root.doc_ids)
        print(root_info)
        _print(self.root, "", 0)

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
            self.root = Node()
            self.word_count = 0
            return

        nodes: List[Node] = [Node() for _ in nodes_data]
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
            self.root = Node()
            self.word_count = 0
            return

        # Reset trie before building
        self.root = Node()
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
    trie = PatriciaTrie()
    trie.load()  # will build from JSONL if the trie JSON doesn't exist
    print(f"Indexed {trie.word_count} unique tokens in the PatriciaTrie inverted index.")
    # Print the Trie structure (edges, end-of-word markers and doc ids)
    #trie.print_tree()
    #print(len(trie.get_all_words()))

if __name__ == "__main__":
    main()