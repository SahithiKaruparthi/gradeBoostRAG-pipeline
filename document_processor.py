#!/usr/bin/env python3
"""
Document Processing Pipeline for FastAPI Backend
Integrates OCR, markdown cleaning, chunking, and database ingestion
"""

import os
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

from pgvector_database import PGVectorDatabase, DocumentChunk
from markdown_cleaner import MarkdownCleaner, CleanedDocument
from smart_chunker import HierarchicalChunker, Chunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing pipeline for uploaded files"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize document processor"""
        self.config = self._load_config(config_path)
        
        # Initialize database
        db_config = self.config.get('database', {})
        self.db = PGVectorDatabase(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'vectorDatabase'),
            user=db_config.get('user', 'postgres'),
            password=db_config.get('password', ''),
            embedding_model=self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Initialize processors
        self.cleaner = MarkdownCleaner()
        self.chunker = HierarchicalChunker(
            textbook_name="Uploaded Document",
            max_chunk_chars=self.config.get('chunk_size', 500),
            min_chunk_chars=self.config.get('min_chunk_size', 100)
        )
        
        logger.info("Document processor initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'chunk_size': 500,
            'chunk_overlap': 100,
            'min_chunk_size': 100,
            'embedding_model': 'all-MiniLM-L6-v2',
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'vectorDatabase',
                'user': 'postgres',
                'password': ''
            }
        }

    def process_document(self, file_path: Path, filename: str) -> Dict[str, Any]:
        """
        Process a document through the complete pipeline:
        1. OCR processing (mock implementation)
        2. Markdown cleaning
        3. Chunking
        4. Database ingestion
        
        Args:
            file_path: Path to the uploaded file
            filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Starting document processing for {filename}")
            
            # Step 1: OCR Processing (Mock implementation)
            # In production, integrate with actual OCR pipeline
            logger.info("Step 1: OCR Processing (Mock)")
            markdown_content = self._mock_ocr_processing(file_path)
            
            # Step 2: Markdown Cleaning
            logger.info("Step 2: Markdown Cleaning")
            cleaned_doc = self._clean_document(markdown_content, filename)
            
            # Step 3: Chunking
            logger.info("Step 3: Document Chunking")
            chunks = self._chunk_document(cleaned_doc)
            
            # Step 4: Create embeddings and database chunks
            logger.info("Step 4: Creating embeddings and database chunks")
            document_chunks = self._create_document_chunks(chunks)
            
            # Step 5: Ingest to database
            logger.info("Step 5: Ingesting to database")
            inserted_count = self._ingest_to_database(document_chunks)
            
            result = {
                'success': True,
                'filename': filename,
                'chunks_created': len(chunks),
                'chunks_inserted': inserted_count,
                'processing_time': datetime.now().isoformat(),
                'message': f"Successfully processed {filename}. {inserted_count} chunks inserted to database."
            }
            
            logger.info(f"Document processing completed for {filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            return {
                'success': False,
                'filename': filename,
                'error': str(e),
                'processing_time': datetime.now().isoformat(),
                'message': f"Failed to process {filename}: {str(e)}"
            }

    def _ocr_processing(self, file_path: Path, filename: str) -> str:
        """
        Process PDF with OCR using Dolphin OCR processor
        In production, integrate with actual OCR pipeline
        
        Args:
            file_path: Path to the PDF file
            filename: Original filename
            
        Returns:
            Markdown content as string
        """
        # Check if file is PDF
        if not filename.lower().endswith('.pdf'):
            logger.warning(f"File {filename} is not a PDF. Using mock OCR.")
            return self._mock_ocr_processing(file_path)
        
        # Try Dolphin OCR if available
        if self.ocr_processor is not None:
            try:
                logger.info(f"Processing {filename} with Dolphin OCR")
                markdown_content = self.ocr_processor.process_pdf(file_path)
                logger.info(f"Dolphin OCR completed for {filename}")
                
                # Optionally save markdown to file for debugging/backup
                temp_md_path = file_path.parent / f"{file_path.stem}.md"
                with open(temp_md_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Markdown saved to {temp_md_path}")
                
                return markdown_content
            except Exception as e:
                logger.error(f"Dolphin OCR failed for {filename}: {e}")
                logger.warning("Falling back to mock OCR")
                return self._mock_ocr_processing(file_path)
        else:
            logger.info(f"OCR processor not available. Using mock OCR for {filename}")
            return self._mock_ocr_processing(file_path)

    def _clean_document(self, content: str, filename: str) -> CleanedDocument:
        """Clean the document content"""
        # Extract metadata
        metadata = self.cleaner.extract_metadata(filename, content)
        
        # Clean the content
        cleaned_content = self.cleaner.clean_text(content)
        
        return CleanedDocument(
            filename=filename,
            content=cleaned_content,
            chapter_number=metadata['chapter_number'],
            page_numbers=metadata['page_numbers'],
            word_count=metadata['word_count'],
            section_count=metadata['section_count']
        )

    def _chunk_document(self, cleaned_doc: CleanedDocument) -> List[Chunk]:
        """Chunk the cleaned document"""
        return self.chunker.chunk_document(cleaned_doc)

    def _create_document_chunks(self, chunks: List[Chunk]) -> List[DocumentChunk]:
        """Create document chunks with embeddings"""
        document_chunks = []
        
        for chunk in chunks:
            try:
                # Generate embedding
                embedding = self.db.generate_embedding(chunk.text)
                
                # Extract section title from header hierarchy
                section_title = ""
                if chunk.header_hierarchy:
                    # Get the most specific header (h3 > h2 > h1)
                    if 'h3' in chunk.header_hierarchy:
                        section_title = chunk.header_hierarchy['h3']
                    elif 'h2' in chunk.header_hierarchy:
                        section_title = chunk.header_hierarchy['h2']
                    elif 'h1' in chunk.header_hierarchy:
                        section_title = chunk.header_hierarchy['h1']
                
                # Create DocumentChunk
                doc_chunk = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=embedding,
                    chapter_number=chunk.chapter_number,
                    section_title=section_title,
                    page_numbers=chunk.page_numbers,
                    word_count=chunk.word_count,
                    source_file=chunk.source_file,
                    chunk_type=chunk.chunk_type,
                    textbook_name="Uploaded Document",
                    created_at=datetime.now().isoformat()
                )
                
                document_chunks.append(doc_chunk)
                
            except Exception as e:
                logger.error(f"Error creating document chunk {chunk.chunk_id}: {e}")
                continue
        
        return document_chunks

    def _ingest_to_database(self, document_chunks: List[DocumentChunk]) -> int:
        """Ingest document chunks into database"""
        if not document_chunks:
            return 0
        
        # Insert in batches for better performance
        batch_size = self.config.get('batch_size', 50)
        total_inserted = 0
        
        for i in range(0, len(document_chunks), batch_size):
            batch = document_chunks[i:i + batch_size]
            inserted = self.db.insert_chunks_batch(batch)
            total_inserted += inserted
            
            logger.info(f"Batch {i//batch_size + 1}: Inserted {inserted}/{len(batch)} chunks")
        
        return total_inserted

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            stats = self.db.get_statistics()
            return {
                'success': True,
                'database_stats': stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()
        logger.info("Document processor closed")

def main():
    """Test the document processor"""
    processor = DocumentProcessor()
    
    # Test with a mock file
    test_file = Path("test_document.txt")
    with open(test_file, 'w') as f:
        f.write("This is a test document for processing.")
    
    try:
        result = processor.process_document(test_file, "test_document.txt")
        print("Processing result:")
        print(json.dumps(result, indent=2))
        
        # Get stats
        stats = processor.get_processing_stats()
        print("\nProcessing stats:")
        print(json.dumps(stats, indent=2))
        
    finally:
        processor.close()
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    main()
