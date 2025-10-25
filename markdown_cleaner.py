#!/usr/bin/env python3
"""
Markdown Cleaner for Biology Textbook Chapters
Cleans OCR artifacts and prepares text for vector database ingestion

The main purpose of this script is to:
    - Clean up common OCR errors like misinterpreted characters, extra spaces, and broken words.
    - Correct domain-specific terminology for biology (e.g., fixing "mitochondri" to "mitochondria").
    - Remove irrelevant artifacts like page numbers, headers, and leftover HTML tags.
    - Extract useful metadata such as chapter number, page numbers, and word count.
    - Save the cleaned text into new files, making them ready for further use, such as being loaded into a vector database for AI applications.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CleanedDocument:
    """Represents a cleaned document with metadata"""
    filename: str
    content: str
    chapter_number: int
    page_numbers: List[int]
    word_count: int
    section_count: int

class MarkdownCleaner:
    """Cleans OCR artifacts from biology textbook markdown files"""
    
    def __init__(self):
        # Common OCR artifacts and their fixes
        self.ocr_fixes = [
            # Chinese characters and artifacts
            (r'[一-龯]+', ''),  # Remove Chinese characters
            (r'[^\x00-\x7F]+', ''),  # Remove non-ASCII characters
            
            # Common OCR mistakes
            (r'\bce\b', 'cell'),
            (r'\bcel\b', 'cell'),
            (r'\buni\b', 'unit'),
            (r'\bchemica\b', 'chemical'),
            
            # Fix broken words
            # (r'(\w+)\s+(\w+)', r'\1\2'),  # Join split words (basic)
            
            # Clean up punctuation
            (r'\s+([,.!?;:])', r'\1'),  # Remove spaces before punctuation
            (r'([,.!?;:])\s*([,.!?;:])', r'\1\2'),  # Fix double punctuation
            
            # Clean up quotes and brackets
            (r'["""]', '"'),      # Normalize quotes
            (r"[']", "'"),        # Normalize apostrophes
            (r'\[([^\]]*)\]', r'\1'),  # Remove square brackets around content
            
            # Clean up figure references
            (r'!\[Figure\]\([^)]*\)', ''),  # Remove broken figure references
            (r'Fig\.\s*\d+\.\d+[^:]*:\s*', ''),  # Remove figure captions
            
            # Clean up table artifacts
            (r'<table[^>]*>.*?</table>', ''),  # Remove HTML table tags
            (r'<tr[^>]*>.*?</tr>', ''),  # Remove table row tags
            (r'<td[^>]*>.*?</td>', ''),  # Remove table cell tags
            (r'rowspan="[^"]*"', ''),  # Remove rowspan attributes
            (r'colspan="[^"]*"', ''),  # Remove colspan attributes
            
            # Clean up page numbers and headers
            (r'^\s*\d+\s*$', ''),  # Remove standalone page numbers
            (r'^\s*CONCISE\s+BIOLOGY[^\n]*$', ''),  # Remove header repetitions
            (r'^\s*[A-Z\s]+\s*$', ''),  # Remove all-caps headers
            
            # Clean up extra whitespace
            (r'\n\s*\n\s*\n+', '\n\n'),  # Reduce multiple newlines to double
            (r'^\s+', ''),  # Remove leading whitespace
            (r'\s+$', ''),  # Remove trailing whitespace
        ]
        
        # Biology-specific term corrections
        self.biology_corrections = [
            (r'\bmitochondri\b', 'mitochondria'),
            (r'\bchloroplast\b', 'chloroplasts'),
            (r'\bchromosom\b', 'chromosome'),
            (r'\bchromatin\b', 'chromatin'),
            (r'\bnucleol\b', 'nucleolus'),
            (r'\bribosom\b', 'ribosomes'),
            (r'\bendoplasmic\s+reticulu\b', 'endoplasmic reticulum'),
            (r'\bcytoplas\b', 'cytoplasm'),
            (r'\bnucleoplas\b', 'nucleoplasm'),
            (r'\bcentromer\b', 'centromere'),
            (r'\bcentrosom\b', 'centrosome'),
            (r'\bchromatid\b', 'chromatids'),
            (r'\bhomozygou\b', 'homozygous'),
            (r'\bheterozygou\b', 'heterozygous'),
            (r'\bgenotyp\b', 'genotype'),
            (r'\bphenotyp\b', 'phenotype'),
            (r'\bhaploid\b', 'haploid'),
            (r'\bdiploid\b', 'diploid'),
            (r'\bmeiosi\b', 'meiosis'),
            (r'\bmitosi\b', 'mitosis'),
            (r'\bkaryokinesi\b', 'karyokinesis'),
            (r'\bcytokinesi\b', 'cytokinesis'),
            (r'\bprophas\b', 'prophase'),
            (r'\bmetaphas\b', 'metaphase'),
            (r'\banaphas\b', 'anaphase'),
            (r'\btelophas\b', 'telophase'),
            (r'\binterphas\b', 'interphase'),
        ]

    def clean_text(self, text: str) -> str:
        """Clean OCR artifacts from text"""
        logger.info("Starting text cleaning...")
        
        # Apply OCR fixes
        for pattern, replacement in self.ocr_fixes:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Apply biology-specific corrections
        for pattern, replacement in self.biology_corrections:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up remaining artifacts
        text = self._clean_remaining_artifacts(text)
        
        logger.info("Text cleaning completed")
        return text

    def _clean_remaining_artifacts(self, text: str) -> str:
        """Clean remaining artifacts that need more complex handling"""
        
        # Remove lines with only Chinese characters or artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that are mostly non-English
            if len(re.sub(r'[a-zA-Z\s]', '', line)) > len(line) * 0.7:
                continue
            
            # Skip very short lines that are likely artifacts
            if len(line.strip()) < 3:
                continue
            
            # Skip lines that are just numbers or special characters
            if re.match(r'^[\d\s\-\.\,]+$', line.strip()):
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def extract_metadata(self, filename: str, content: str) -> Dict:
        """Extract metadata from filename and content"""
        
        # Extract chapter number from filename
        chapter_match = re.search(r'chapter\s+(\d+)', filename, re.IGNORECASE)
        chapter_number = int(chapter_match.group(1)) if chapter_match else 0
        
        # Extract page numbers from content
        page_numbers = re.findall(r'(\d+)\s*$', content, re.MULTILINE)
        page_numbers = [int(p) for p in page_numbers if p.isdigit()]
        
        # Count sections
        section_count = len(re.findall(r'^#+\s+', content, re.MULTILINE))
        
        # Count words
        word_count = len(re.findall(r'\b\w+\b', content))
        
        return {
            'chapter_number': chapter_number,
            'page_numbers': sorted(list(set(page_numbers))),
            'section_count': section_count,
            'word_count': word_count,
            'filename': filename
        }

    def clean_document(self, filepath: Path) -> CleanedDocument:
        """Clean a single markdown document"""
        logger.info(f"Cleaning document: {filepath}")
        
        # Read the file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean the content
        cleaned_content = self.clean_text(content)
        
        # Extract metadata
        metadata = self.extract_metadata(filepath.name, cleaned_content)
        
        return CleanedDocument(
            filename=filepath.name,
            content=cleaned_content,
            chapter_number=metadata['chapter_number'],
            page_numbers=metadata['page_numbers'],
            word_count=metadata['word_count'],
            section_count=metadata['section_count']
        )

    def clean_all_documents(self, input_dir: Path) -> List[CleanedDocument]:
        """Clean all markdown documents in a directory"""
        logger.info(f"Cleaning all documents in: {input_dir}")
        
        markdown_files = list(input_dir.glob("*.md"))
        cleaned_documents = []
        
        for filepath in markdown_files:
            try:
                cleaned_doc = self.clean_document(filepath)
                cleaned_documents.append(cleaned_doc)
                logger.info(f"Cleaned: {filepath.name} - {cleaned_doc.word_count} words, {cleaned_doc.section_count} sections")
            except Exception as e:
                logger.error(f"Error cleaning {filepath}: {e}")
        
        return cleaned_documents

    def save_cleaned_documents(self, cleaned_docs: List[CleanedDocument], output_dir: Path):
        """Save cleaned documents to output directory"""
        output_dir.mkdir(exist_ok=True)
        
        for doc in cleaned_docs:
            output_path = output_dir / f"cleaned_{doc.filename}"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc.content)
            logger.info(f"Saved cleaned document: {output_path}")

def main():
    """Main function to clean all markdown files"""
    input_dir = Path("markdown")
    output_dir = Path("markdown_cleaned")
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    cleaner = MarkdownCleaner()
    cleaned_docs = cleaner.clean_all_documents(input_dir)
    
    if cleaned_docs:
        cleaner.save_cleaned_documents(cleaned_docs, output_dir)
        
        # Print summary
        total_words = sum(doc.word_count for doc in cleaned_docs)
        total_sections = sum(doc.section_count for doc in cleaned_docs)
        
        print(f"\n=== CLEANING SUMMARY ===")
        print(f"Documents cleaned: {len(cleaned_docs)}")
        print(f"Total words: {total_words:,}")
        print(f"Total sections: {total_sections}")
        print(f"Output directory: {output_dir}")
        
        for doc in cleaned_docs:
            print(f"  - {doc.filename}: {doc.word_count:,} words, {doc.section_count} sections")

if __name__ == "__main__":
    main()
