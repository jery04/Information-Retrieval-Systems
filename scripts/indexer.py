import unicodedata
import re
from typing import List, Optional, Set
import spacy

# Conjunto de stop words en español (comúnmente usadas)
# Usado en `tokenize()` para filtrar palabras vacías adicionales a las de spaCy
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
    """
    Clase para preprocesamiento de texto en español.

    Métodos:
    - normalize(word): minusculas, quitar acentos, quitar puntuación simple
    - is_stopword(word): verifica stop words
    - reduce(word, method): reduce por 'stem' o 'lemmatize'
    - process(texts, reduction): realiza una sola pasada sobre el array
      recibido, tokeniza por espacios, normaliza, filtra stopwords,
      reduce y acumula tokens resultantes.
    """
    def __init__(self):
        # Cargar y cachear el modelo spaCy
        self.nlp = spacy.load("es_core_news_sm")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizador general usando spaCy.

        - Lematiza, pasa a minúsculas, quita acentos y filtra puntuación/espacios.
        - Filtra stopwords usando `token.is_stop` y la lista `self.stop_words`.
        """
        tokens: List[str] = []
        if not text:
            return tokens

        doc = self.nlp(text)
        for token in doc:
            # Filtramos puntuación y espacios
            if token.is_punct or token.is_space:
                continue
            # Lematizamos y pasamos a minúsculas
            word = token.lemma_.lower()
            # Quitamos acentos
            word = unicodedata.normalize("NFKD", word)
            word = "".join(ch for ch in word if not unicodedata.combining(ch))
            # Filtrar tokens cortos, no alfabéticos o stopwords
            if len(word) <= 1:
                continue
            if not word.isalpha():
                continue
            if token.is_stop or (word in stop_words):
                continue
            tokens.append(word)

        return tokens

class TrieNode:
    def __init__(self):
        self.children = {}           # type: dict[str, TrieNode]
        self.is_end_of_word = False  # ¿Aquí termina una palabra?
        self.doc_ids = []            # Lista de doc_id donde aparece esta palabra

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, doc_id: Optional[int] = None) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end_of_word = True
        if doc_id is not None:
            if doc_id not in node.doc_ids:
                node.doc_ids.append(doc_id)

    def search(self, word: str) -> Optional[TrieNode]:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node if node.is_end_of_word else None

    def starts_with(self, prefix: str) -> List[str]:
        results: List[str] = []

        def _collect(n: TrieNode, path: str):
            if n.is_end_of_word:
                results.append(path)
            for c, child in n.children.items():
                _collect(child, path + c)

        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        _collect(node, prefix)
        return results

if __name__ == "__main__":
    # Pequeña demostración (cada llamada procesa un solo string)
    index = Index()
    sample_text = "¡Hola! Corriendo y comiendo, esto es una prueba de tokenización."
    print(index.tokenize(sample_text))