"""
A_Ingest.py - read the PDF books, build the OKF bundle and the vector database.

Run it once (and again whenever you add or change a book):
    python A_Ingest.py

  STEP 1  read_book()        PDF -> sections. The PDF bookmarks (table of contents)
                             tell us the chapter and section of every page.
  STEP 2  build_okf()        sections -> OKF bundle: one Markdown file per book,
                             chapter and section, linked together (okf_bundle/).
  STEP 3  build_vector_db()  sections -> small overlapping chunks -> embeddings
                             saved in ChromaDB, each with book, section and page.
"""

import os
import re
import shutil
from datetime import datetime, timezone

from google_okf import Concept, validate_bundle_links, write_bundle
from pypdf import PdfReader

import B_rag as settings
from B_rag import create_embedder, get_client, make_aliases

# Bookmarks with these names are not study material, so we skip them
SKIP = re.compile(r"^([\d.]+\s+)?(preface|foreword|acknowledg|contents|index|references|"
                  r"bibliography|exercises|problems|further reading|notes|notation)", re.IGNORECASE)


def slugify(text):
    """'2.4 Vector Spaces' -> 'vector-spaces'  (for file names)"""
    text = re.sub(r"^[\d.]+\s+", "", text.lower())
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:40] or "untitled"


def book_info(filename):
    """'mml-book.pdf' -> ('MML', 'Mathematics for Machine Learning')"""
    for key, (short_name, title) in settings.BOOK_NAMES.items():
        if key in filename:
            return short_name, title
    name = filename.replace(".pdf", "")
    return name, name


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 - PDF -> sections
# ═════════════════════════════════════════════════════════════════════════════

def get_bookmarks(reader):
    """The PDF bookmarks as a list of {title, page, is_chapter}, sorted by page.
    Top-level bookmarks are chapters. 'Part I' bookmarks are skipped, so the
    chapters inside a part also count as top level."""
    bookmarks = []

    def walk(items, depth):
        previous_was_part = False
        for item in items:
            if isinstance(item, list):                 # a list = sub-bookmarks of the one before
                walk(item, depth if previous_was_part else depth + 1)
                continue
            title = item.title.replace("\xa0", " ").strip()
            previous_was_part = title.lower().startswith("part ")
            if not previous_was_part:
                bookmarks.append({"title": title,
                                  "page": reader.get_destination_page_number(item) + 1,
                                  "is_chapter": depth == 0})

    walk(reader.outline, 0)
    return sorted(bookmarks, key=lambda b: b["page"])


def read_book(pdf_path):
    """Return the book as a list of sections:
    {"chapter": "2 Linear Algebra", "title": "2.5 Linear Independence", "pages": [(46, text), ...]}"""
    reader = PdfReader(pdf_path)
    bookmarks = get_bookmarks(reader)

    sections = []
    chapter = section = None
    next_bookmark = 0
    for number, page in enumerate(reader.pages, start=1):
        # Move forward through the bookmarks that start on or before this page
        while next_bookmark < len(bookmarks) and bookmarks[next_bookmark]["page"] <= number:
            bookmark = bookmarks[next_bookmark]
            if bookmark["is_chapter"]:
                chapter = bookmark["title"]
            section = bookmark["title"]
            next_bookmark += 1

        if chapter is None or SKIP.match(chapter) or SKIP.match(section):
            continue
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if len(text) < 200:                            # skip empty / picture-only pages
            continue

        if sections and sections[-1]["title"] == section and sections[-1]["chapter"] == chapter:
            sections[-1]["pages"].append((number, text))
        else:
            sections.append({"chapter": chapter, "title": section, "pages": [(number, text)]})
    return sections


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 - sections -> OKF bundle (linked Markdown files)
# ═════════════════════════════════════════════════════════════════════════════

def safe_markdown(text):
    """PDF text can look like Markdown links by accident; break those up."""
    return text.replace("](", "] (")


def build_okf(books):
    """books: list of {"short", "title", "file", "sections"}.
    Writes okf_bundle/<book>/chNN-<chapter>/sNN-<section>.md and adds
    section["concept_id"] to every section (used later by ChromaDB)."""
    now = datetime.now(timezone.utc)
    concepts = {}

    for book in books:
        book_id = slugify(book["short"])
        resource = f"pdfs/{book['file']}"

        # Group the sections by chapter, keeping the book's order
        chapters = {}
        for section in book["sections"]:
            chapters.setdefault(section["chapter"], []).append(section)

        chapter_links = []
        for chapter_number, (chapter_title, sections) in enumerate(chapters.items(), start=1):
            chapter_dir = f"{book_id}/ch{chapter_number:02d}-{slugify(chapter_title)}"
            file_names = [f"s{n:02d}-{slugify(s['title'])}" for n, s in enumerate(sections, start=1)]

            for i, section in enumerate(sections):
                concept_id = f"{chapter_dir}/{file_names[i]}"
                section["concept_id"] = concept_id
                first_page, last_page = section["pages"][0][0], section["pages"][-1][0]
                text = "\n\n".join(t for _, t in section["pages"])

                # Links to the chapter and to the previous / next section
                navigation = [f"Up: [{chapter_title}](index.md)"]
                neighbours = []
                if i > 0:
                    navigation.append(f"Previous: [{sections[i - 1]['title']}]({file_names[i - 1]}.md)")
                    neighbours.append(f"{chapter_dir}/{file_names[i - 1]}")
                if i + 1 < len(sections):
                    navigation.append(f"Next: [{sections[i + 1]['title']}]({file_names[i + 1]}.md)")
                    neighbours.append(f"{chapter_dir}/{file_names[i + 1]}")

                aliases = make_aliases(re.sub(r"^[\d.]+\s+", "", section["title"]))

                concepts[concept_id] = Concept(
                    type="section",
                    title=section["title"],
                    description=text[:200] + "…",
                    resource=f"{resource}#page={first_page}",
                    tags=["ai", book_id],
                    timestamp=now,
                    body=f"# {section['title']}\n\n" + "  \n".join(navigation) + "\n\n" + safe_markdown(text) + "\n",
                    book=book["short"], chapter=chapter_title, pages=[first_page, last_page],
                    aliases=aliases, neighbours=neighbours,
                )

            concepts[f"{chapter_dir}/index"] = Concept(
                type="chapter",
                title=chapter_title,
                description=f"Chapter of {book['title']} with {len(sections)} sections.",
                resource=f"{resource}#page={sections[0]['pages'][0][0]}",
                tags=["ai", book_id], timestamp=now,
                body=f"# {chapter_title}\n\nUp: [{book['title']}](../index.md)\n\n## Sections\n\n"
                     + "\n".join(f"- [{s['title']}]({f}.md)" for s, f in zip(sections, file_names)) + "\n",
                book=book["short"],
            )
            chapter_links.append(f"- [{chapter_title}]({chapter_dir.split('/', 1)[1]}/index.md)")

        concepts[f"{book_id}/index"] = Concept(
            type="document", title=book["title"],
            description=f"{book['short']}: {len(chapters)} chapters, {len(book['sections'])} sections.",
            resource=resource, tags=["ai", book_id], timestamp=now,
            body=f"# {book['title']}\n\nUp: [All books](../index.md)\n\n## Chapters\n\n" + "\n".join(chapter_links) + "\n",
            book=book["short"],
        )

    concepts["index"] = Concept(
        type="index", title="AI books", description="The books used by the AI Concepts Explainer.",
        tags=["ai"], timestamp=now,
        body="# AI books\n\n" + "\n".join(f"- [{b['title']}]({slugify(b['short'])}/index.md)" for b in books) + "\n",
    )

    if os.path.exists(settings.OKF_DIR):
        shutil.rmtree(settings.OKF_DIR)
    write_bundle(settings.OKF_DIR, concepts)

    broken = validate_bundle_links(settings.OKF_DIR)
    print(f"OKF bundle: {len(concepts)} files written to {settings.OKF_DIR}"
          + (f" ({len(broken)} broken links)" if broken else " (all links work)"))


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 - sections -> chunks -> ChromaDB
# ═════════════════════════════════════════════════════════════════════════════

def split_into_chunks(text, size, overlap):
    """Cut text into pieces of about `size` characters, without cutting words."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_vector_db(books):
    if os.path.exists(settings.CHROMA_DIR):
        shutil.rmtree(settings.CHROMA_DIR)

    collection = get_client().create_collection(
        name=settings.COLLECTION,
        embedding_function=create_embedder(),
        metadata={"hnsw:space": "cosine"},     # compare chunks by cosine similarity
    )

    for book in books:
        ids, texts, metadatas = [], [], []
        for section in book["sections"]:
            for page_number, text in section["pages"]:
                for i, chunk in enumerate(split_into_chunks(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)):
                    ids.append(f"{book['short']}-p{page_number}-{i}")
                    texts.append(f"{section['title']}\n{chunk}")    # the title helps the search
                    metadatas.append({"book": book["short"], "section": section["title"],
                                      "page": page_number, "concept_id": section["concept_id"]})

        for start in range(0, len(ids), 500):                      # ChromaDB likes batches
            collection.add(ids=ids[start:start + 500],
                           documents=texts[start:start + 500],
                           metadatas=metadatas[start:start + 500])
        print(f"ChromaDB: {book['short']} -> {len(ids)} chunks")

    print(f"ChromaDB: {collection.count()} chunks saved in {settings.CHROMA_DIR}")


# ═════════════════════════════════════════════════════════════════════════════

def main():
    pdf_files = sorted(f for f in os.listdir(settings.PDF_DIR) if f.endswith(".pdf"))
    if not pdf_files:
        print(f"No PDFs found in the '{settings.PDF_DIR}' folder.")
        return

    books = []
    for filename in pdf_files:
        short_name, title = book_info(filename)
        sections = read_book(os.path.join(settings.PDF_DIR, filename))
        chapters = len({s["chapter"] for s in sections})
        print(f"{short_name}: {chapters} chapters, {len(sections)} sections")
        books.append({"short": short_name, "title": title, "file": filename, "sections": sections})

    build_okf(books)
    build_vector_db(books)
    print("Done. Start the server: uvicorn main:app --reload")


if __name__ == "__main__":
    main()