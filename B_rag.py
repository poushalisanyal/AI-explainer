"""
B_rag.py - settings + searching the books.
(Retrieval-Augmented Generation: this file is the "Retrieval" part.)

Two ways to find the right part of a book:

  1. OKF exact lookup   The question names a section of a book, e.g.
                        "What is linear independence?" -> MML section 2.5.
                        We read that whole section from the OKF bundle. Score = 1.0.
  2. Vector search      Otherwise, ChromaDB finds the chunks closest in meaning.

OKF (Open Knowledge Format) = the books saved as linked Markdown files:
book -> chapter -> section, built by A_Ingest.py in the okf_bundle/ folder.

A_Ingest.py imports the settings below, so both files always match.

Test:  python B_rag.py
"""

import os
import re

import chromadb
from chromadb.utils import embedding_functions
from google_okf import read_bundle

# ── Settings ──────────────────────────────────────────────────────────────────

PDF_DIR = "./pdfs"
OKF_DIR = "./okf_bundle"
CHROMA_DIR = "./chroma_db"
COLLECTION = "ai_books"
EMBED_MODEL = "all-MiniLM-L6-v2"   # free model that runs on your own computer

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared by two neighbouring chunks
TOP_K = 4              # how many chunks are given to the LLM
MIN_SCORE = 0.35       # below this similarity the books are not used
MAX_SECTION_CHARS = 4000   # how much of an exact-match section is given to the LLM

# File name (or part of it) -> (short name for citations, full title)
BOOK_NAMES = {
    "mml-book": ("MML", "Mathematics for Machine Learning"),
    "UnderstandingDeepLearning": ("UDL", "Understanding Deep Learning"),
    "978-981-92-0682-7": ("NLP-LLM", "Natural Language Processing and Large Language Models"),
}


# ── Turning titles and questions into lookup keys ─────────────────────────────

QUESTION_START = re.compile(
    r"^(what\s+is|what\s+are|what's|explain|define|meaning\s+of|how\s+does|how\s+do|"
    r"tell\s+me\s+about|introduction\s+to|basics\s+of|overview\s+of)\s+", re.IGNORECASE)

# Section titles too general to be looked up ("What is introduction?" makes no sense)
TOO_GENERAL = {"introduction", "overview", "summary", "conclusion", "background", "motivation",
               "example", "experiment", "discussion", "notation", "terminology", "problem",
               "application", "implementation", "result", "preliminarie", "model", "data"}


def term_key(text):
    """'What are Word Embeddings?' -> 'word embedding'   (used to compare terms)"""
    text = re.sub(r"^[\d.]+\s+", "", text.strip().lower())    # "2.4 vector spaces" -> "vector spaces"
    text = QUESTION_START.sub("", text)
    text = re.sub(r"^(an?|the)\s+", "", text)
    text = text.strip(" ?.!\"'")
    words = text.split()
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]                             # plural -> singular
    return " ".join(words)


def make_aliases(title):
    """Lookup keys for a section title.
    'Long-Short Term Memory, LSTM'        -> ['long-short term memory', 'lstm']
    'Principal Component Analysis (PCA)'  -> ['principal component analysi', 'pca']"""
    names = re.findall(r"\(([A-Za-z\- ]{2,15})\)", title)      # abbreviations in brackets
    base = re.sub(r"\s*\([^)]*\)", "", title)
    if ", " in base and base.rsplit(", ", 1)[1].isupper():     # "Name, ABBR"
        base, abbreviation = base.rsplit(", ", 1)
        names.append(abbreviation)
    aliases = []
    for name in [base] + names:
        key = term_key(name)
        if len(key) > 2 and key not in TOO_GENERAL and key not in aliases:
            aliases.append(key)
    return aliases


# ── Database helpers (also used by A_Ingest.py) ───────────────────────────────

def get_client():
    """Open (or create) the ChromaDB database on disk."""
    return chromadb.PersistentClient(path=CHROMA_DIR)


def create_embedder():
    """The model that turns text into a list of numbers (an embedding)."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


# ── 1. OKF exact lookup ───────────────────────────────────────────────────────

class OKFIndex:
    def __init__(self):
        self.sections = {}    # concept id -> section
        self.by_alias = {}    # "lstm" -> concept id
        if not os.path.isdir(OKF_DIR):
            print("OKF bundle not found. Run: python A_Ingest.py")
            return
        for concept_id, concept in sorted(read_bundle(OKF_DIR).items()):
            if concept.type == "section":
                self.sections[concept_id] = concept
                for alias in concept.model_extra.get("aliases", []):
                    self.by_alias.setdefault(alias, concept_id)

    def related_topics(self, concept_id):
        """Titles of the previous and next section in the same chapter."""
        concept = self.sections.get(concept_id)
        if not concept:
            return []
        titles = []
        for other_id in concept.model_extra.get("neighbours", []):
            if other_id in self.sections:
                title = re.sub(r"^[\d.]+\s+", "", self.sections[other_id].title)
                if title not in titles:
                    titles.append(title)
        return titles[:5]

    def lookup(self, question):
        """Return the whole matching section, or None if no section has this name."""
        concept_id = self.by_alias.get(term_key(question))
        if not concept_id:
            return None
        concept = self.sections[concept_id]
        text = re.sub(r"^(Up|Previous|Next):.*$", "", concept.body, flags=re.MULTILINE).strip()
        return {
            "text": text[:MAX_SECTION_CHARS],
            "book": concept.model_extra["book"],
            "section": concept.title,
            "page": concept.model_extra["pages"][0],
            "score": 1.0,
            "found_by": "okf",
            "related": self.related_topics(concept_id),
        }


# ── 2. Vector search + putting both together ──────────────────────────────────

class BookSearch:
    def __init__(self):
        self.okf = OKFIndex()
        self.collection = get_client().get_collection(
            name=COLLECTION,
            embedding_function=create_embedder(),
        )

    def search(self, question, top_k=TOP_K):
        """Best matching book passages, best first, each with a score from 0 to 1."""
        exact_match = self.okf.lookup(question)
        if exact_match:
            return [exact_match]

        result = self.collection.query(query_texts=[question], n_results=top_k)
        matches = []
        for text, meta, distance in zip(result["documents"][0],
                                        result["metadatas"][0],
                                        result["distances"][0]):
            matches.append({
                "text": text,
                "book": meta["book"],
                "section": meta["section"],
                "page": meta["page"],
                "score": round(1 - distance, 3),     # cosine distance -> similarity
                "found_by": "vector",
                "related": self.okf.related_topics(meta["concept_id"]),
            })
        return matches


if __name__ == "__main__":
    books = BookSearch()
    print(f"OKF: {len(books.okf.sections)} sections, {len(books.okf.by_alias)} terms. "
          f"ChromaDB: {books.collection.count()} chunks.")
    for question in ["What is linear independence?", "what is LSTM", "What is backpropagation?",
                     "Who won the IPL?"]:
        best = books.search(question)[0]
        print(f"{question:30} -> {best['found_by']:6} {best['book']}, p. {best['page']} "
              f"score={best['score']}  related={best['related'][:3]}")