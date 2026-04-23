"""Lightweight indexing utilities: Spanish tokenizer and PatriciaTrie inverted index.

This module provides simple helpers for Spanish text preprocessing and a
PatriciaTrie-based inverted index with JSON persistence. Docstrings are brief and
focused on functionality (main() is intentionally unchanged).
"""

import unicodedata  # Tools for Unicode normalization (remove accents, etc.)
import json         # Read and write JSON data
import os           # Interact with the operating system (paths, files)
from typing import Any, Dict, List, Optional, Set  # Type hints for cleaner code
from file_read_backwards import FileReadBackwards  # Read large files from bottom to top efficiently
import spacy        # NLP library for tokenization, lemmatization, etc.

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
        self.document_count = 0
        # default path if not provided
        if filepath is None:
            self.filepath = os.path.join("data", "processed", "inverted_index_trie.json")
        else:
            self.filepath = filepath

    @staticmethod
    def _parse_doc_id(raw_doc_id: Any) -> Optional[int]:
        """Convert a raw doc id to int; return None if conversion fails."""
        try:
            return int(raw_doc_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_webpages_jsonl_path() -> Optional[str]:
        """Return the first available JSONL source path for webpages."""
        candidates = [
            os.path.join("data", "extracted", "webpages", "webpages.jsonl"),
            os.path.join("data", "extracterd", "webpages", "webpages.jsonl"),
            os.path.join("data", "extracted", "webpages", "sample_webpages.jsonl"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _insert_document_from_json(self, obj: Dict[str, Any]) -> Optional[int]:
        """Insert one JSONL document into the trie and return its integer doc_id."""
        doc_id = self._parse_doc_id(obj.get("doc_id"))
        if doc_id is None:
            return None

        text = obj.get("text", "")
        tokens = Index.tokenize(text)
        for token in set(tokens):
            self.insert(token, doc_id)

        # Keep this as the highest ingested document id.
        if doc_id > self.document_count:
            self.document_count = doc_id

        return doc_id

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

        doc_id_int = self._parse_doc_id(doc_id)
        if doc_id_int is not None and doc_id_int > self.document_count:
            self.document_count = doc_id_int

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
                    if doc_id_int is not None and doc_id_int not in split_node.doc_ids:
                        split_node.doc_ids.append(doc_id_int)
                    return

                # Add a new branch for the remaining suffix.
                new_leaf = Node()
                new_leaf.is_end_of_word = True
                if doc_id_int is not None:
                    new_leaf.doc_ids.append(doc_id_int)
                split_node.children[new_suffix] = new_leaf
                self.word_count += 1
                return

            if not matched:
                # No shared prefix from this node: create a direct compressed edge.
                new_leaf = Node()
                new_leaf.is_end_of_word = True
                if doc_id_int is not None:
                    new_leaf.doc_ids.append(doc_id_int)
                node.children[remaining] = new_leaf
                self.word_count += 1
                return

        # Word already represented by the current node path.
        if not node.is_end_of_word:
            node.is_end_of_word = True
            self.word_count += 1
        if doc_id_int is not None and doc_id_int not in node.doc_ids:
            node.doc_ids.append(doc_id_int)

    def search(self, word: str) -> Optional[List[int]]:
        """Return list of document IDs for exact `word`, or None if not found."""
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

        return node.doc_ids if node.is_end_of_word else None

    def intersect_tokens(self, tokens: List[str]) -> List[int]:
        """Return the intersection of doc_ids for the given list of tokens.

        Calls `search` for each token; if any token is not found, returns
        an empty list (no documents contain all tokens). Result is returned
        as a sorted list of integers.
        """
        # Defensive: empty input -> no results
        if not tokens:
            return []

        sets: List[Set[int]] = []
        for t in tokens:
            docs = self.search(t)
            # if token not present or has no docs -> intersection empty
            if not docs:
                return []
            sets.append(set(docs))

        # sort by size for faster intersection
        sets.sort(key=len)

        result = sets[0]
        for s in sets[1:]:
            result &= s
            if not result:
                return []

        return sorted(result)

    def get_parcial_AND(self, query_terms: List[str], min_match: int = 2, max_candidates: int = 3000) -> List[int]:
        """Partial-AND

        Returns a list of candidate `doc_id` values that contain at least
        `min_match` terms from `query_terms`. Limits the number of returned
        results to `max_candidates`, sorted by the number of matches
        (highest first).
        """
        if not query_terms:
            return []

        # 1. Obtener posting lists y ordenarlas por longitud (de menor a mayor)
        term_postings = []
        for term in set(query_terms):
            docs = self.search(term)
            if docs:
                docs_sorted = sorted(docs)   # importante: ordenados
                term_postings.append((len(docs_sorted), docs_sorted, term))

        if len(term_postings) < min_match:
            return []

        # Ordenar por frecuencia ascendente (más raro primero)
        term_postings.sort()

        # Usamos un contador por documento (dict)
        from collections import defaultdict
        doc_count = defaultdict(int)

        # Procesamos cada posting list
        for _, posting_list, _ in term_postings:
            for doc_id in posting_list:
                doc_count[doc_id] += 1

        # Filtramos los documentos que cumplen el mínimo
        candidates = [doc_id for doc_id, count in doc_count.items() 
                      if count >= min_match]

        # Ordenamos por cantidad de matches (mejor primero)
        candidates.sort(key=lambda d: doc_count[d], reverse=True)

        return candidates[:max_candidates]

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

        # Include word and document counters for persistence
        return {
            "nodes": nodes,
            "root": 0,
            "count": self.word_count,
            "doc_count": self.document_count,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Reconstruct the Trie from serialized `data` and apply to self.

        If `data` contains no nodes, the trie is reset to an empty state.
        """
        nodes_data = data.get("nodes", [])

        if not nodes_data:
            self.root = Node()
            self.word_count = 0
            self.document_count = 0
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

        persisted_doc_count = data.get("doc_count")
        if isinstance(persisted_doc_count, int) and persisted_doc_count >= 0:
            self.document_count = persisted_doc_count
        else:
            max_doc_id = 0
            for node in nodes:
                for raw_doc_id in node.doc_ids:
                    parsed = self._parse_doc_id(raw_doc_id)
                    if parsed is not None and parsed > max_doc_id:
                        max_doc_id = parsed
            self.document_count = max_doc_id
        return

    def save(self) -> None:
        """Save the Trie to a JSON file at `filepath` (or default path)."""

        # Ensure the directory exists
        dirpath = os.path.dirname(self.filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(self.filepath, "w", encoding="utf-8") as f:
            # Use compact separators to save space and speed I/O
            json.dump(self.to_dict(), f, ensure_ascii=False, separators=(",", ":"))

    def load(self) -> None:
        """Load persisted trie and incrementally sync from latest JSONL lines.

        Behavior:
        1) If the trie JSON exists, load it.
        2) Independently of that, read webpages JSONL from end to start.
        3) Compare latest `doc_id` with current `document_count`.
        4) If they differ, ingest only the newest documents (reverse order)
           until reaching the line whose `doc_id` equals the initial count.
        """

        trie_file_exists = os.path.exists(self.filepath)

        if trie_file_exists:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.from_dict(data)
        else:
            self.root = Node()
            self.word_count = 0
            self.document_count = 0

        initial_document_count = self.document_count
        webpages_path = self._resolve_webpages_jsonl_path()
        if not webpages_path:
            if not trie_file_exists:
                self.save()
            return

        latest_doc_id: Optional[int] = None
        with FileReadBackwards(webpages_path, encoding="utf-8") as frb:
            for raw_line in frb:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                parsed_doc_id = self._parse_doc_id(obj.get("doc_id"))
                if parsed_doc_id is not None:
                    latest_doc_id = parsed_doc_id
                    break

        if latest_doc_id is None or latest_doc_id == initial_document_count:
            return

        updated = False
        with FileReadBackwards(webpages_path, encoding="utf-8") as frb:
            for raw_line in frb:
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = self._parse_doc_id(obj.get("doc_id"))
                if doc_id is None:
                    continue

                if doc_id == initial_document_count:
                    break
                if doc_id < initial_document_count:
                    break

                inserted_doc_id = self._insert_document_from_json(obj)
                if inserted_doc_id is not None:
                    updated = True

        if updated:
            self.save()

def main():
    trie = PatriciaTrie()
    trie.load()  # will build from JSONL if the trie JSON doesn't exist
    print(f"Indexed {trie.word_count} unique tokens in the PatriciaTrie inverted index.")

if __name__ == "__main__":
    main()