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
    def __init__(self):
        """Load and cache the small Spanish spaCy model."""
        self.nlp = spacy.load("es_core_news_sm")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize `text` and return a list of cleaned tokens.

        Steps: lemmatize, lowercase, strip accents, remove punctuation/space
        tokens, filter out short/non-alpha tokens and stopwords.
        """
        tokens: List[str] = []
        if not text:
            return tokens

        doc = self.nlp(text)
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
        """Load the Trie from a JSON file and apply it to this instance."""
        if filepath is None:
            filepath = self.filepath

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)

def main():
    index = Index()
    trie = Trie()

    documents = [
        (1, "¡Hola! Corriendo y hablando en una prueba de tokenización."),
        (2, "Juguetes, panes, comidas, y manzanas... El perro corre rápido mientras come en el corral."),
        (3, "Prueba de búsqueda con palabras en español."),
        (4, "La corrida matutina es parte de la rutina diaria."),
        (5, "Los niños juegan en el parque y corren alrededor."),
        (6, "Un gato duerme sobre el sofá durante la tarde."),
        (7, "La cocina tiene un olor delicioso a pan recién hecho."),
        (8, "Investigación y desarrollo: sistemas de recuperación de información."),
        (9, "Búsqueda por prefijo y por término exacto en estructuras tipo trie."),
        (10, "Pruebas adicionales con acentos: rápido, corré, acción, corazón.")
    ]
    i=0
    for doc_id, text in documents:
        tokens = index.tokenize(text)
        for token in set(tokens):           # set() para no repetir en el mismo documento
            trie.insert(token, doc_id)
            i+=1
    print("SIUUUUUUUUUUUUUUUUUUUUUUU",i)
    print(trie.word_count, "palabras únicas indexadas")
    # Guardar el índice y medir tiempo
    t0 = time.time()
    trie.save()
    t_save = time.time() - t0
    print(f"Índice guardado correctamente en {t_save:.4f}s")

    # Tamaño del archivo
    try:
        size = os.path.getsize(os.path.join("data", "processed", "inverted_index_trie.json"))
        print(f"Tamaño archivo: {size} bytes")
    except Exception:
        pass

    # Cargar en otro momento (al iniciar el programa) y medir tiempo
    t0 = time.time()
    loaded_trie = Trie()
    loaded_trie.load()
    t_load = time.time() - t0
    print(f"Índice cargado correctamente en {t_load:.4f}s")
    print(loaded_trie.word_count, "palabras únicas indexadas")

    # Pruebas: búsquedas y prefijos
    queries = ["correr", "prueba", "perro", "corral", "hola", "informacion"]
    for q in queries:
        node = loaded_trie.search(q)
        if node:
            print(f"Documentos con '{q}':", node.doc_ids)
        else:
            print(f"No se encontró '{q}' como término exacto")

if __name__ == "__main__":
    main()