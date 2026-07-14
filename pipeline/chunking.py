"""
Structure-aware, parent-child chunking -- mirrors section 16.1 of your
Guidewire interview doc:
  1. Split on Markdown headers first (preserves section hierarchy)
  2. Split oversized sections into 1024-token parents
  3. Split each parent into 256-token children (these get embedded/searched)

Pure logic, no external model calls -- fully testable offline.
"""

import json
import os
import glob
from dataclasses import dataclass, asdict

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

# NOTE: chunk_size here is in *characters* via RecursiveCharacterTextSplitter,
# not tokens. ~4 chars/token is a reasonable rule of thumb, so 1024 tokens ~=
# 4000 chars, 256 tokens ~= 1000 chars. Scaled down proportionally is fine too
# -- what matters for the interview story is the parent:child ratio (4:1),
# not the literal number, so keep them ~4:1 whatever absolute size you use.
PARENT_SIZE = 1024 * 4
PARENT_OVERLAP = 100 * 4
CHILD_SIZE = 256 * 4
CHILD_OVERLAP = 32 * 4

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_SIZE, chunk_overlap=PARENT_OVERLAP)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_SIZE, chunk_overlap=CHILD_OVERLAP)


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    text: str
    is_parent: bool
    metadata: dict


def chunk_document(markdown_text: str, doc_meta: dict) -> list[Chunk]:
    """Returns a flat list of Chunk objects: parents AND children.
    Children carry parent_id so ParentDocumentRetriever-style lookup works."""

    sections = header_splitter.split_text(markdown_text)
    all_chunks = []
    parent_counter = 0

    for section in sections:
        section_text = section.page_content
        header_meta = section.metadata  # e.g. {"h1": "...", "h2": "...", "h3": "..."}

        parents = parent_splitter.split_text(section_text) or [section_text]

        for p_text in parents:
            parent_counter += 1
            parent_id = f"{doc_meta['doc_id']}_p{parent_counter}"

            parent_meta = {**doc_meta, **header_meta}
            all_chunks.append(Chunk(
                chunk_id=parent_id,
                parent_id=parent_id,
                text=p_text,
                is_parent=True,
                metadata=parent_meta,
            ))

            children = child_splitter.split_text(p_text)
            for c_idx, c_text in enumerate(children):
                child_id = f"{parent_id}_c{c_idx}"
                all_chunks.append(Chunk(
                    chunk_id=child_id,
                    parent_id=parent_id,
                    text=c_text,
                    is_parent=False,
                    metadata=parent_meta,
                ))

    return all_chunks


def build_chunk_store():
    """Walk the synthetic corpus, chunk every doc, write chunks.jsonl.
    This file is what the embedding step (run on your laptop) will consume."""

    metadata_path = os.path.join(DATA_DIR, "corpus_metadata.json")
    with open(metadata_path) as f:
        corpus_meta = json.load(f)

    out_path = os.path.join(DATA_DIR, "chunks.jsonl")
    total_chunks, total_children = 0, 0

    with open(out_path, "w") as out_f:
        for doc_meta in corpus_meta:
            file_path = os.path.join(DATA_DIR, "raw_docs", doc_meta["file"])
            with open(file_path) as f:
                markdown_text = f.read()

            chunks = chunk_document(markdown_text, doc_meta)
            for c in chunks:
                out_f.write(json.dumps(asdict(c)) + "\n")
                total_chunks += 1
                if not c.is_parent:
                    total_children += 1

    print(f"Wrote {total_chunks} chunks ({total_children} children, "
          f"{total_chunks - total_children} parents) to {out_path}")


if __name__ == "__main__":
    build_chunk_store()
