"""
FastAPI Backend for RAG Pipeline
Handles document upload, OCR processing, chunking, and MCQ generation
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

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import existing modules
from rag_pipeline_json import RAGPipelineJSON, MCQuestion, MCQOption, RAGPipelineResult
from pgvector_database import PGVectorDatabase, DocumentChunk
from document_processor import DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fastapi_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="RAG pipeline for document processing and MCQ generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for pipeline components
rag_pipeline = None
document_processor = None
db = None

# Pydantic models for API
class MCQGenerationRequest(BaseModel):
    topic: str = Field(..., description="Topic for MCQ generation")
    difficulty: str = Field(default="medium", description="Difficulty level: easy, medium, hard")
    chapter: Optional[int] = Field(None, description="Optional chapter filter")
    num_questions: int = Field(default=5, description="Number of questions to generate")

class MCQResponse(BaseModel):
    success: bool
    topic: str
    mcqs: List[Dict[str, Any]]
    retrieval_time: float
    generation_time: float
    total_time: float
    timestamp: str
    error_message: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    message: str
    processing_status: str
    timestamp: str

class DocumentProcessingStatus(BaseModel):
    document_id: str
    status: str  # uploaded, processing, completed, failed
    progress: int  # 0-100
    message: str
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

class DatabaseStats(BaseModel):
    total_chunks: int
    total_words: int
    chunks_by_chapter: Dict[str, int]
    chunks_by_type: Dict[str, int]
    source_files: Dict[str, int]
    embedding_model: str
    embedding_dimension: int

# In-memory storage for processing status (use Redis in production)
processing_status = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline and database connections"""
    global rag_pipeline, document_processor, db
    
    try:
        # Load configuration
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Initialize database
        db_config = config.get('database', {})
        db = PGVectorDatabase(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'vectorDatabase'),
            user=db_config.get('user', 'postgres'),
            password=db_config.get('password', ''),
            embedding_model=config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipelineJSON('config.json')
        
        # Initialize document processor
        document_processor = DocumentProcessor('config.json')
        
        logger.info("FastAPI backend initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    global db, rag_pipeline, document_processor
    
    if db:
        db.close()
    if rag_pipeline:
        rag_pipeline.close()
    if document_processor:
        document_processor.close()
    
    logger.info("FastAPI backend shutdown completed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        stats = db.get_statistics()
        return {
            "status": "healthy",
            "database": "connected",
            "total_chunks": stats.get('total_chunks', 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a document for processing (OCR, cleaning, chunking, and database ingestion)
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate unique document ID
    document_id = str(uuid.uuid4())
    
    try:
        # Create temporary directory for processing
        temp_dir = Path(f"temp_processing/{document_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize processing status
        processing_status[document_id] = DocumentProcessingStatus(
            document_id=document_id,
            status="uploaded",
            progress=0,
            message="Document uploaded successfully",
            created_at=datetime.now().isoformat()
        )
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            document_id,
            file_path,
            file.filename
        )
        
        return DocumentUploadResponse(
            success=True,
            document_id=document_id,
            filename=file.filename,
            message="Document uploaded successfully. Processing started.",
            processing_status="processing",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

async def process_document_background(document_id: str, file_path: Path, filename: str):
    """Background task to process uploaded document"""
    try:
        # Update status
        processing_status[document_id].status = "processing"
        processing_status[document_id].progress = 10
        processing_status[document_id].message = "Starting document processing..."
        
        # Use the document processor to handle the complete pipeline
        result = document_processor.process_document(file_path, filename)
        
        if result['success']:
            # Update final status
            processing_status[document_id].status = "completed"
            processing_status[document_id].progress = 100
            processing_status[document_id].message = result['message']
            processing_status[document_id].completed_at = datetime.now().isoformat()
            
            logger.info(f"Document processing completed for {filename}")
        else:
            # Update failed status
            processing_status[document_id].status = "failed"
            processing_status[document_id].error_message = result.get('error', 'Unknown error')
            processing_status[document_id].message = result['message']
            
            logger.error(f"Document processing failed for {filename}: {result.get('error', 'Unknown error')}")
        
        # Clean up temporary files
        try:
            shutil.rmtree(file_path.parent)
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up temporary files: {cleanup_error}")
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        processing_status[document_id].status = "failed"
        processing_status[document_id].error_message = str(e)
        processing_status[document_id].message = f"Document processing failed: {str(e)}"

@app.get("/document-status/{document_id}", response_model=DocumentProcessingStatus)
async def get_document_status(document_id: str):
    """Get the processing status of a document"""
    if document_id not in processing_status:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return processing_status[document_id]

@app.post("/generate-mcqs", response_model=MCQResponse)
async def generate_mcqs(request: MCQGenerationRequest):
    """
    Generate MCQs for a given topic using the RAG pipeline
    """
    try:
        logger.info(f"Generating MCQs for topic: {request.topic}")
        
        # Generate MCQs using the RAG pipeline
        result = rag_pipeline.generate_mcqs(
            topic=request.topic,
            difficulty=request.difficulty,
            chapter=request.chapter
        )
        
        if not result.success:
            return MCQResponse(
                success=False,
                topic=request.topic,
                mcqs=[],
                retrieval_time=result.retrieval_time,
                generation_time=result.generation_time,
                total_time=result.total_time,
                timestamp=datetime.now().isoformat(),
                error_message=result.error_message
            )
        
        # Convert MCQs to structured JSON format
        mcqs_json = []
        for mcq in result.mcqs:
            mcq_data = {
                "question_id": str(uuid.uuid4()),
                "question_text": mcq.question_text,
                "options": [
                    {
                        "letter": opt.letter,
                        "text": opt.text
                    } for opt in mcq.options
                ],
                "correct_answer": mcq.correct_answer,
                "explanation": mcq.explanation,
                "difficulty": mcq.difficulty,
                "topic": mcq.topic,
                "source_chunks": mcq.source_chunks,
                "created_at": datetime.now().isoformat()
            }
            mcqs_json.append(mcq_data)
        
        # Store MCQs in database
        await store_mcqs_in_database(mcqs_json, request.topic)
        
        return MCQResponse(
            success=True,
            topic=request.topic,
            mcqs=mcqs_json,
            retrieval_time=result.retrieval_time,
            generation_time=result.generation_time,
            total_time=result.total_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating MCQs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate MCQs: {str(e)}")

async def store_mcqs_in_database(mcqs: List[Dict], topic: str):
    """Store generated MCQs in the database"""
    try:
        with db.connection.cursor() as cursor:
            for mcq in mcqs:
                cursor.execute("""
                    INSERT INTO generated_mcqs (
                        mcq_id, topic, question_text, options, correct_answer,
                        explanation, difficulty, source_chunks, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    mcq["question_id"],
                    topic,
                    mcq["question_text"],
                    json.dumps(mcq["options"]),
                    mcq["correct_answer"],
                    mcq["explanation"],
                    mcq["difficulty"],
                    json.dumps(mcq["source_chunks"]),
                    mcq["created_at"]
                ))
        
        logger.info(f"Stored {len(mcqs)} MCQs in database for topic: {topic}")
        
    except Exception as e:
        logger.error(f"Error storing MCQs in database: {e}")
        # Don't raise exception here to avoid breaking the API response

@app.get("/database-stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get database statistics"""
    try:
        stats = db.get_statistics()
        return DatabaseStats(**stats)
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get database stats: {str(e)}")

@app.get("/mcqs/{topic}")
async def get_mcqs_by_topic(topic: str, limit: int = 10):
    """Get stored MCQs for a specific topic"""
    try:
        with db.connection.cursor() as cursor:
            cursor.execute("""
                SELECT mcq_id, topic, question_text, options, correct_answer,
                       explanation, difficulty, source_chunks, created_at
                FROM generated_mcqs
                WHERE topic ILIKE %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (f"%{topic}%", limit))
            
            results = cursor.fetchall()
            
            mcqs = []
            for row in results:
                mcq = {
                    "question_id": row[0],
                    "topic": row[1],
                    "question_text": row[2],
                    "options": json.loads(row[3]),
                    "correct_answer": row[4],
                    "explanation": row[5],
                    "difficulty": row[6],
                    "source_chunks": json.loads(row[7]),
                    "created_at": row[8].isoformat() if row[8] else None
                }
                mcqs.append(mcq)
            
            return {
                "success": True,
                "topic": topic,
                "mcqs": mcqs,
                "count": len(mcqs),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting MCQs for topic {topic}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get MCQs: {str(e)}")

@app.delete("/mcqs/{question_id}")
async def delete_mcq(question_id: str):
    """Delete a specific MCQ"""
    try:
        with db.connection.cursor() as cursor:
            cursor.execute("DELETE FROM generated_mcqs WHERE mcq_id = %s", (question_id,))
            
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="MCQ not found")
            
            return {
                "success": True,
                "message": f"MCQ {question_id} deleted successfully",
                "timestamp": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting MCQ {question_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete MCQ: {str(e)}")

if __name__ == "__main__":
    # Create the MCQ storage table if it doesn't exist
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        db_config = config.get('database', {})
        temp_db = PGVectorDatabase(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'vectorDatabase'),
            user=db_config.get('user', 'postgres'),
            password=db_config.get('password', ''),
            embedding_model=config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Create MCQ storage table
        with temp_db.connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generated_mcqs (
                    id SERIAL PRIMARY KEY,
                    mcq_id VARCHAR(255) UNIQUE NOT NULL,
                    topic VARCHAR(255) NOT NULL,
                    question_text TEXT NOT NULL,
                    options JSONB NOT NULL,
                    correct_answer VARCHAR(10) NOT NULL,
                    explanation TEXT,
                    difficulty VARCHAR(20),
                    source_chunks JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mcq_topic 
                ON generated_mcqs (topic);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mcq_difficulty 
                ON generated_mcqs (difficulty);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mcq_created_at 
                ON generated_mcqs (created_at);
            """)
        
        temp_db.close()
        logger.info("MCQ storage table created successfully")
        
    except Exception as e:
        logger.error(f"Error creating MCQ storage table: {e}")
    
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
