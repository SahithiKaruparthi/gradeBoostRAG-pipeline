
#!/usr/bin/env python3
"""
Hierarchical Chunker for Biology Textbook Content.
Reads pre-cleaned markdown files and creates context-aware chunks.

Its main job is to take the cleaned markdown files from the previous step and break them down into smaller, intelligent pieces, or "chunks."

The goal is to create chunks that are not only small enough for an AI to process but also contain rich contextual metadata. 
This is a crucial step for building a high-quality Retrieval-Augmented Generation (RAG) system, as the AI will know exactly 
where each piece of information came from (e.g., "Chapter 5, Section 'The Cell', Subsection 'Mitochondria'").

"""

import re
import logging
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field

# We still need these classes from the cleaner script, but we won't run the cleaning process again.
from markdown_cleaner import CleanedDocument, MarkdownCleaner

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Dataclass for Chunks ---
@dataclass
class Chunk:
    """Represents a text chunk with rich metadata for RAG pipelines"""
    text: str
    chunk_id: str
    source_file: str
    textbook_name: str
    chapter_number: int
    header_hierarchy: Dict[str, str] = field(default_factory=dict)
    page_numbers: List[int] = field(default_factory=list)
    word_count: int = 0
    chunk_type: str = 'paragraph'


# --- Chunker Class ---
class HierarchicalChunker:
    """Chunks markdown documents by respecting their hierarchical structure (headers)."""

    def __init__(self,
                 textbook_name: str,
                 max_chunk_chars: int = 1500,
                 min_chunk_chars: int = 250):
        self.textbook_name = textbook_name
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.header_patterns = {
            1: re.compile(r'^#\s+(.*)'),
            2: re.compile(r'^##\s+(.*)'),
            3: re.compile(r'^###\s+(.*)'),
        }

    def chunk_document(self, doc: CleanedDocument) -> List[Chunk]:
        """Main method to chunk a single cleaned document."""
        logger.info(f"Starting hierarchical chunking for {doc.filename}...")
        
        h2_sections = re.split(r'(^##\s+.*)', doc.content, flags=re.MULTILINE)
        
        all_chunks = []
        chunk_counter = 0
        
        intro_content = h2_sections[0].strip()
        if intro_content:
            all_chunks.extend(self._process_section(
                intro_content,
                {'h1': f"Chapter {doc.chapter_number} Introduction"},
                doc,
                chunk_counter
            ))
            chunk_counter = len(all_chunks)

        for i in range(1, len(h2_sections), 2):
            header_line = h2_sections[i]
            section_content = h2_sections[i+1]
            
            h2_title_match = self.header_patterns[2].match(header_line)
            if not h2_title_match: continue
                
            h2_title = h2_title_match.group(1).strip()
            current_hierarchy = {'h1': f"Chapter {doc.chapter_number}", 'h2': h2_title}
            
            all_chunks.extend(self._process_section(
                section_content,
                current_hierarchy,
                doc,
                chunk_counter
            ))
            chunk_counter = len(all_chunks)
            
        final_chunks = self._merge_small_chunks(all_chunks)
        logger.info(f"Completed chunking for {doc.filename}. Final chunks: {len(final_chunks)}")
        return final_chunks

    def _process_section(self, section_content: str, hierarchy: Dict[str, str], doc: CleanedDocument, start_chunk_id: int) -> List[Chunk]:
        """Processes content within a section, splitting by the next header level."""
        chunks = []
        
        h3_sections = re.split(r'(^###\s+.*)', section_content, flags=re.MULTILINE)
        
        intro_content = h3_sections[0].strip()
        if intro_content:
            chunks.extend(self._chunk_prose(intro_content, hierarchy, doc, start_chunk_id))
        
        for i in range(1, len(h3_sections), 2):
            header_line = h3_sections[i]
            subsection_content = h3_sections[i+1]
            
            h3_title_match = self.header_patterns[3].match(header_line)
            if not h3_title_match: continue

            h3_title = h3_title_match.group(1).strip()
            sub_hierarchy = {**hierarchy, 'h3': h3_title}
            
            chunks.extend(self._chunk_prose(subsection_content, sub_hierarchy, doc, len(chunks) + start_chunk_id))

        return chunks

    def _chunk_prose(self, text: str, hierarchy: Dict[str, str], doc: CleanedDocument, start_chunk_id: int) -> List[Chunk]:
        """Chunks plain text by splitting into paragraphs/lists and handling size constraints."""
        base_chunks_text = re.split(r'\n\s*\n', text.strip())
        
        chunks = []
        chunk_counter = 0

        for content in base_chunks_text:
            content = content.strip()
            if not content: continue
            
            if len(content) > self.max_chunk_chars:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                current_split_chunk = ""
                for sentence in sentences:
                    if len(current_split_chunk) + len(sentence) + 1 > self.max_chunk_chars:
                        if current_split_chunk:
                            chunks.append(self._create_chunk_object(current_split_chunk, hierarchy, doc, start_chunk_id + chunk_counter))
                            chunk_counter += 1
                        current_split_chunk = sentence
                    else:
                        current_split_chunk += (" " + sentence) if current_split_chunk else sentence
                
                if current_split_chunk:
                    chunks.append(self._create_chunk_object(current_split_chunk, hierarchy, doc, start_chunk_id + chunk_counter))
                    chunk_counter += 1
            else:
                chunks.append(self._create_chunk_object(content, hierarchy, doc, start_chunk_id + chunk_counter))
                chunk_counter += 1
        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merges consecutive small chunks under the same header."""
        if not chunks: return []
            
        merged_chunks = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            if (current_chunk.header_hierarchy == next_chunk.header_hierarchy and
                len(current_chunk.text) + len(next_chunk.text) < self.max_chunk_chars):
                current_chunk.text += "\n\n" + next_chunk.text
                current_chunk.word_count += next_chunk.word_count
                current_chunk.page_numbers = sorted(list(set(current_chunk.page_numbers + next_chunk.page_numbers)))
            else:
                if len(current_chunk.text) >= self.min_chunk_chars:
                    merged_chunks.append(current_chunk)
                current_chunk = next_chunk
        
        if len(current_chunk.text) >= self.min_chunk_chars:
            merged_chunks.append(current_chunk)
            
        for i, chunk in enumerate(merged_chunks):
            base_name = chunk.source_file.replace('.md', '')
            chunk.chunk_id = f"{base_name}_chunk_{i:04d}"

        return merged_chunks

    def _create_chunk_object(self, text: str, hierarchy: Dict, doc: CleanedDocument, chunk_id: int) -> Chunk:
        """Helper to create a Chunk dataclass object with all metadata."""
        base_name = doc.filename.replace('.md', '')
        return Chunk(
            text=text,
            chunk_id=f"{base_name}_chunk_{chunk_id:04d}",
            source_file=doc.filename,
            textbook_name=self.textbook_name,
            chapter_number=doc.chapter_number,
            header_hierarchy=hierarchy,
            page_numbers=doc.page_numbers,
            word_count=len(re.findall(r'\b\w+\b', text)),
            chunk_type=self._determine_chunk_type(text)
        )

    def _determine_chunk_type(self, text: str) -> str:
        """Determines chunk type based on simple content heuristics."""
        if re.match(r'^\s*[-*+]\s+', text, flags=re.MULTILINE):
            return 'list'
        if re.match(r'^\s*\d+\.\s+', text, flags=re.MULTILINE):
            return 'numbered_list'
        return 'paragraph'

# --- Main Execution Logic ---
def main():
    """
    Main function to load cleaned files and run the hierarchical chunking process.
    """
    # --- Configuration ---
    CLEANED_FILES_DIR = Path("markdown_cleaned")
    CHUNK_OUTPUT_DIR = Path("chunks_hierarchical")
    TEXTBOOK_NAME = "Class X Biology NCERT"

    CHUNK_OUTPUT_DIR.mkdir(exist_ok=True)

    # --- 1. LOAD PRE-CLEANED DOCUMENTS ---
    logger.info(f"Loading pre-cleaned documents from '{CLEANED_FILES_DIR}'...")
    
    # We use an instance of the cleaner ONLY to access its helper method
    # for extracting metadata. No re-cleaning is performed.
    metadata_helper = MarkdownCleaner()
    docs_to_chunk = []

    if not CLEANED_FILES_DIR.exists() or not any(CLEANED_FILES_DIR.iterdir()):
        logger.error(f"Input directory '{CLEANED_FILES_DIR}' is empty or does not exist.")
        return
        
    for filepath in CLEANED_FILES_DIR.glob("*.md"):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata from the already-clean content
        metadata = metadata_helper.extract_metadata(filepath.name, content)
        
        # Create the document object required by the chunker
        doc = CleanedDocument(content=content, **metadata)
        docs_to_chunk.append(doc)

    logger.info(f"Successfully loaded {len(docs_to_chunk)} documents.")

    # --- 2. CHUNK THE DOCUMENTS ---
    logger.info("Starting chunking process...")
    chunker = HierarchicalChunker(
        textbook_name=TEXTBOOK_NAME,
        max_chunk_chars=1200,
        min_chunk_chars=150
    )
    
    all_chunks = []
    for doc in docs_to_chunk:
        all_chunks.extend(chunker.chunk_document(doc))
        
    # --- 3. SAVE CHUNKS AND REPORT ---
    if all_chunks:
        for chunk in all_chunks:
            filepath = CHUNK_OUTPUT_DIR / f"{chunk.chunk_id}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"CHUNK METADATA:\n")
                f.write(f"  ID: {chunk.chunk_id}\n")
                f.write(f"  Textbook: {chunk.textbook_name}\n")
                f.write(f"  Source File: {chunk.source_file}\n")
                f.write(f"  Chapter: {chunk.chapter_number}\n")
                f.write(f"  Pages: {chunk.page_numbers}\n")
                f.write(f"  Hierarchy: {chunk.header_hierarchy}\n")
                f.write("-------------------- CONTENT --------------------\n\n")
                f.write(chunk.text)
        
        print("\n=== HIERARCHICAL CHUNKING SUMMARY ===")
        print(f"Textbook: '{TEXTBOOK_NAME}'")
        print(f"Total documents processed: {len(docs_to_chunk)}")
        print(f"Total chunks created: {len(all_chunks)}")
        avg_words = sum(c.word_count for c in all_chunks) / len(all_chunks)
        print(f"Average words per chunk: {avg_words:.2f}")
        print(f"Chunks saved to: '{CHUNK_OUTPUT_DIR}'")

if __name__ == "__main__":
    main()