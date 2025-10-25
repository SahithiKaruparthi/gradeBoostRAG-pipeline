#!/usr/bin/env python3
"""
PGVector Database Implementation for RAG Pipeline
Handles vector storage, retrieval, and metadata management

This script creates and manages a Vector Database using PostgreSQL and the pgvector extension. 
Its purpose is to take the text chunks from the previous script, convert them into numerical representations called embeddings (or vectors), 
and store everything in a database.

The key function of this database is to perform semantic search. 
Instead of searching for keywords, it searches for meaning. When you ask a question, it finds the chunks of text from the textbook 
that are most conceptually similar to your question. This is the "Retrieval" part of RAG.
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a document chunk with vector and metadata"""
    chunk_id: str
    text: str
    embedding: List[float]
    chapter_number: int
    section_title: str
    page_numbers: List[int]
    word_count: int
    source_file: str
    chunk_type: str
    textbook_name: str = "Class X Biology"
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

class PGVectorDatabase:
    """PGVector database implementation for biology textbook chunks"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "vectorDatabase",
                 user: str = "postgres",
                 password: str = "",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.embedding_model_name = embedding_model
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {embedding_model}, dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Database connection
        self.connection = None
        self._connect()
        self._setup_database()

    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.connection.autocommit = True
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _setup_database(self):
        """Setup database schema and extensions"""
        try:
            with self.connection.cursor() as cursor:
                # Enable pgvector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        chunk_id VARCHAR(255) UNIQUE NOT NULL,
                        text TEXT NOT NULL,
                        embedding vector(%d) NOT NULL,
                        chapter_number INTEGER NOT NULL,
                        section_title VARCHAR(500),
                        page_numbers INTEGER[],
                        word_count INTEGER NOT NULL,
                        source_file VARCHAR(255) NOT NULL,
                        chunk_type VARCHAR(50),
                        textbook_name VARCHAR(255),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                """ % self.embedding_dimension)
                
                # Create indexes for better performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_embedding 
                    ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_chapter 
                    ON document_chunks (chapter_number);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_source 
                    ON document_chunks (source_file);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_type 
                    ON document_chunks (chunk_type);
                """)
                
                logger.info("Database schema setup completed")
                
        except Exception as e:
            logger.error(f"Failed to setup database schema: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def insert_chunk(self, chunk: DocumentChunk) -> bool:
        """Insert a single chunk into the database"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO document_chunks (
                        chunk_id, text, embedding, chapter_number, section_title,
                        page_numbers, word_count, source_file, chunk_type, textbook_name
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON CONFLICT (chunk_id) 
                    DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        chapter_number = EXCLUDED.chapter_number,
                        section_title = EXCLUDED.section_title,
                        page_numbers = EXCLUDED.page_numbers,
                        word_count = EXCLUDED.word_count,
                        source_file = EXCLUDED.source_file,
                        chunk_type = EXCLUDED.chunk_type,
                        textbook_name = EXCLUDED.textbook_name,
                        updated_at = NOW();
                """, (
                    chunk.chunk_id,
                    chunk.text,
                    chunk.embedding,
                    chunk.chapter_number,
                    chunk.section_title,
                    chunk.page_numbers,
                    chunk.word_count,
                    chunk.source_file,
                    chunk.chunk_type,
                    chunk.textbook_name
                ))
                
                logger.debug(f"Inserted chunk: {chunk.chunk_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert chunk {chunk.chunk_id}: {e}")
            return False

    def insert_chunks_batch(self, chunks: List[DocumentChunk]) -> int:
        """Insert multiple chunks in batch"""
        successful_inserts = 0
        
        try:
            with self.connection.cursor() as cursor:
                for chunk in chunks:
                    try:
                        cursor.execute("""
                            INSERT INTO document_chunks (
                                chunk_id, text, embedding, chapter_number, section_title,
                                page_numbers, word_count, source_file, chunk_type, textbook_name
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                            ) ON CONFLICT (chunk_id) 
                            DO UPDATE SET
                                text = EXCLUDED.text,
                                embedding = EXCLUDED.embedding,
                                chapter_number = EXCLUDED.chapter_number,
                                section_title = EXCLUDED.section_title,
                                page_numbers = EXCLUDED.page_numbers,
                                word_count = EXCLUDED.word_count,
                                source_file = EXCLUDED.source_file,
                                chunk_type = EXCLUDED.chunk_type,
                                textbook_name = EXCLUDED.textbook_name,
                                updated_at = NOW();
                        """, (
                            chunk.chunk_id,
                            chunk.text,
                            chunk.embedding,
                            chunk.chapter_number,
                            chunk.section_title,
                            chunk.page_numbers,
                            chunk.word_count,
                            chunk.source_file,
                            chunk.chunk_type,
                            chunk.textbook_name
                        ))
                        successful_inserts += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to insert chunk {chunk.chunk_id}: {e}")
                        continue
                        
            logger.info(f"Successfully inserted {successful_inserts}/{len(chunks)} chunks")
            return successful_inserts
            
        except Exception as e:
            logger.error(f"Failed batch insert: {e}")
            return successful_inserts

    def search_similar(self, query: str, top_k: int = 5, 
                      filters: Optional[Dict] = None) -> List[Dict]:
        """Search for similar chunks using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Build SQL query with filters
            base_query = """
                SELECT 
                    chunk_id, text, chapter_number, section_title, page_numbers,
                    word_count, source_file, chunk_type, textbook_name,
                    1 - (embedding <=> %s::vector) as similarity_score
                FROM document_chunks
                WHERE 1=1
            """
            
            params = [query_embedding]
            
            # Add filters
            if filters:
                if 'chapter_number' in filters:
                    base_query += " AND chapter_number = %s"
                    params.append(filters['chapter_number'])
                
                if 'chunk_type' in filters:
                    base_query += " AND chunk_type = %s"
                    params.append(filters['chunk_type'])
                
                if 'source_file' in filters:
                    base_query += " AND source_file = %s"
                    params.append(filters['source_file'])
            
            base_query += " ORDER BY embedding <=> %s::vector LIMIT %s;"
            params.extend([query_embedding, top_k])
            
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(base_query, params)
                results = cursor.fetchall()
                
                # Convert to list of dicts
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Failed to search similar chunks: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Get a specific chunk by ID"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM document_chunks WHERE chunk_id = %s;
                """, (chunk_id,))
                
                result = cursor.fetchone()
                return dict(result) if result else None
                
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

    def get_statistics(self) -> Dict:
        """Get database statistics"""
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Total chunks
                cursor.execute("SELECT COUNT(*) as total_chunks FROM document_chunks;")
                total_chunks = cursor.fetchone()['total_chunks']
                
                # Chunks by chapter
                cursor.execute("""
                    SELECT chapter_number, COUNT(*) as count 
                    FROM document_chunks 
                    GROUP BY chapter_number 
                    ORDER BY chapter_number;
                """)
                chunks_by_chapter = dict(cursor.fetchall())
                
                # Chunks by type
                cursor.execute("""
                    SELECT chunk_type, COUNT(*) as count 
                    FROM document_chunks 
                    GROUP BY chunk_type 
                    ORDER BY count DESC;
                """)
                chunks_by_type = dict(cursor.fetchall())
                
                # Total words
                cursor.execute("SELECT SUM(word_count) as total_words FROM document_chunks;")
                total_words = cursor.fetchone()['total_words'] or 0
                
                # Source files
                cursor.execute("""
                    SELECT source_file, COUNT(*) as count 
                    FROM document_chunks 
                    GROUP BY source_file 
                    ORDER BY source_file;
                """)
                source_files = dict(cursor.fetchall())
                
                return {
                    'total_chunks': total_chunks,
                    'total_words': total_words,
                    'chunks_by_chapter': chunks_by_chapter,
                    'chunks_by_type': chunks_by_type,
                    'source_files': source_files,
                    'embedding_model': self.embedding_model_name,
                    'embedding_dimension': self.embedding_dimension
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def delete_chunks_by_source(self, source_file: str) -> int:
        """Delete all chunks from a specific source file"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM document_chunks WHERE source_file = %s;
                """, (source_file,))
                
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} chunks from {source_file}")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to delete chunks from {source_file}: {e}")
            return 0

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

def main():
    """Test the PGVector database"""
    # This is a test function - you'll need to configure your database connection
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'biology_rag',
        'user': 'postgres',
        'password': 'your_password_here'
    }
    
    try:
        # Initialize database
        db = PGVectorDatabase(**db_config)
        
        # Test statistics
        stats = db.get_statistics()
        print("Database Statistics:")
        print(json.dumps(stats, indent=2))
        
        # Test search
        results = db.search_similar("What is a cell?", top_k=3)
        print(f"\nSearch results for 'What is a cell?':")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['chunk_id']} (similarity: {result['similarity_score']:.3f})")
            print(f"   Chapter {result['chapter_number']}: {result['section_title']}")
            print(f"   {result['text'][:100]}...")
            print()
        
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

