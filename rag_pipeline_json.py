#!/usr/bin/env python3
"""
Production-Grade RAG Pipeline for Biology Textbook MCQ Generation with JSON Output
Uses PGVector for retrieval and Groq API for LLM-based MCQ generation
Returns structured JSON format for MCQs
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import sys

from pgvector_database import PGVectorDatabase, DocumentChunk
from groq import Groq

# Setup logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/rag_pipeline_json.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# === DATA MODELS ===
# =============================================================================

@dataclass
class MCQOption:
    """Represents a single MCQ option"""
    letter: str  # A, B, C, D
    text: str

@dataclass
class MCQuestion:
    """Represents a complete multiple choice question"""
    question_text: str
    options: List[MCQOption]
    correct_answer: str  # A, B, C, or D
    explanation: str
    difficulty: str  # easy, medium, hard
    topic: str
    source_chunks: List[Dict]  # Metadata about source chunks
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'question': self.question_text,
            'options': [{'letter': opt.letter, 'text': opt.text} for opt in self.options],
            'correct_answer': self.correct_answer,
            'explanation': self.explanation,
            'difficulty': self.difficulty,
            'topic': self.topic,
            'sources': self.source_chunks
        }

@dataclass
class RAGPipelineResult:
    """Result of RAG pipeline execution"""
    success: bool
    topic: str
    mcqs: List[MCQuestion]
    retrieval_time: float
    generation_time: float
    total_time: float
    error_message: Optional[str] = None
    database_stats: Optional[Dict] = None

# =============================================================================
# === MCQ GENERATOR CLASS ===
# =============================================================================

class MCQGeneratorJSON:
    """Generates MCQs using Groq API based on retrieved context with JSON output"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Groq client"""
        api_key = api_key or os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not provided. Set it as environment variable or pass it directly."
            )
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        logger.info(f"Initialized MCQ Generator with model: {self.model}")

    def generate_mcqs(self, context: str, topic: str, difficulty: str = "medium") -> List[MCQuestion]:
        """
        Generate 5 MCQs based on context using Groq API with JSON output
        
        Args:
            context: Combined text from retrieved chunks
            topic: The topic for MCQ generation
            difficulty: Difficulty level (easy, medium, hard)
        
        Returns:
            List of MCQuestion objects
        """
        logger.info(f"Generating {difficulty} MCQs for topic: {topic}")
        
        prompt = self._build_json_prompt(context, topic, difficulty)
        
        try:
            # Call Groq API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert educator specializing in biology education. "
                            "Generate high-quality, clear, and factually accurate multiple-choice questions. "
                            "Each question should test conceptual understanding, not just memorization. "
                            "You MUST respond with valid JSON format only."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
                top_p=0.9
            )
            
            response_text = completion.choices[0].message.content.strip()
            logger.info("Successfully received MCQs from Groq API")
            
            # Parse the JSON response
            mcqs = self._parse_json_mcq_response(response_text, topic)
            return mcqs
            
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise

    def _build_json_prompt(self, context: str, topic: str, difficulty: str) -> str:
        return f"""Based on the following educational content about "{topic}", generate exactly 5 multiple-choice questions (MCQs) at {difficulty} difficulty level.

IMPORTANT: You MUST respond with valid JSON format only. No additional text or explanations.

Required JSON format:
{{
  "mcqs": [
    {{
      "question": "Question text here?",
      "options": [
        {{"letter": "A", "text": "Option A text"}},
        {{"letter": "B", "text": "Option B text"}},
        {{"letter": "C", "text": "Option C text"}},
        {{"letter": "D", "text": "Option D text"}}
      ],
      "correct_answer": "B",
      "explanation": "Brief explanation of why the correct answer is right",
      "difficulty": "{difficulty}",
      "topic": "{topic}"
    }}
  ]
}}

Requirements:
- Generate exactly 5 MCQs
- Each MCQ must have exactly 4 options (A, B, C, D)
- Questions should test conceptual understanding, not memorization
- Ensure questions are clear and unambiguous
- Do NOT repeat concepts across questions
- Make sure the JSON is valid and properly formatted

---EDUCATIONAL CONTENT---
{context}

---END CONTENT---

Generate exactly 5 MCQs at {difficulty} level for the topic "{topic}". Respond with valid JSON only."""

    def _parse_json_mcq_response(self, response_text: str, topic: str) -> List[MCQuestion]:
        """
        Parse JSON MCQ response from LLM into MCQuestion objects
        """
        import re
        
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove any markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Parse JSON
            data = json.loads(response_text)
            
            if 'mcqs' not in data:
                raise ValueError("No 'mcqs' key found in response")
            
            mcqs = []
            for mcq_data in data['mcqs']:
                try:
                    # Extract question text
                    question_text = mcq_data.get('question', '')
                    if not question_text:
                        continue
                    
                    # Extract options
                    options = []
                    for opt_data in mcq_data.get('options', []):
                        letter = opt_data.get('letter', '')
                        text = opt_data.get('text', '')
                        if letter and text:
                            options.append(MCQOption(letter=letter, text=text))
                    
                    if len(options) < 4:
                        logger.warning(f"Question has fewer than 4 options, skipping: {question_text}")
                        continue
                    
                    # Extract other fields
                    correct_answer = mcq_data.get('correct_answer', 'A')
                    explanation = mcq_data.get('explanation', '')
                    difficulty = mcq_data.get('difficulty', 'medium')
                    
                    mcq = MCQuestion(
                        question_text=question_text,
                        options=options,
                        correct_answer=correct_answer,
                        explanation=explanation,
                        difficulty=difficulty,
                        topic=topic,
                        source_chunks=[]  # Will be populated by RAGPipeline
                    )
                    mcqs.append(mcq)
                    
                except Exception as e:
                    logger.warning(f"Error parsing MCQ: {e}")
                    continue
            
            if not mcqs:
                logger.warning("No MCQs were successfully parsed from JSON response")
            
            return mcqs
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Error parsing JSON MCQ response: {e}")
            return []

# =============================================================================
# === RAG PIPELINE CLASS ===
# =============================================================================

class RAGPipelineJSON:
    """Production-grade RAG pipeline for MCQ generation with JSON output"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize RAG pipeline"""
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
        
        # Initialize MCQ generator
        self.mcq_generator = MCQGeneratorJSON()
        
        # RAG configuration
        self.rag_config = self.config.get('rag_pipeline', {})
        self.top_k = self.rag_config.get('top_k', 5)
        self.max_context_length = self.rag_config.get('max_context_length', 8000)
        self.similarity_threshold = self.rag_config.get('similarity_threshold', 0.5)
        
        logger.info("RAG Pipeline JSON initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
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
            'embedding_model': 'all-MiniLM-L6-v2',
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'vectorDatabase',
                'user': 'postgres',
                'password': ''
            },
            'rag_pipeline': {
                'top_k': 5,
                'max_context_length': 8000,
                'similarity_threshold': 0.5
            }
        }

    def retrieve_chunks(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant chunks from database"""
        import time
        start = time.time()
        
        logger.info(f"Retrieving chunks for query: {query}")
        
        try:
            results = self.db.search_similar(query, top_k=self.top_k, filters=filters)
            
            # Filter by similarity threshold
            filtered = [r for r in results if r.get('similarity_score', 0) >= self.similarity_threshold]
            
            retrieval_time = time.time() - start
            logger.info(f"Retrieved {len(filtered)} chunks in {retrieval_time:.2f}s")
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

    def format_context(self, chunks: List[Dict]) -> Tuple[str, List[Dict]]:
        """Format chunks into context for LLM"""
        context_parts = []
        source_metadata = []
        
        for i, chunk in enumerate(chunks):
            # Create source reference
            source_info = {
                'reference': f"[{i+1}]",
                'chunk_id': chunk.get('chunk_id', 'N/A'),
                'chapter': chunk.get('chapter_number', 'N/A'),
                'section': chunk.get('section_title', 'N/A'),
                'page_numbers': chunk.get('page_numbers', []),
                'source_file': chunk.get('source_file', 'N/A'),
                'similarity_score': round(chunk.get('similarity_score', 0), 3),
                'chunk_type': chunk.get('chunk_type', 'N/A')
            }
            source_metadata.append(source_info)
            
            # Format chunk with reference
            chunk_text = (
                f"[{i+1}] Chapter {chunk.get('chapter_number', 'N/A')}: "
                f"{chunk.get('section_title', 'N/A')}\n"
                f"Source: {chunk.get('source_file', 'N/A')}\n"
                f"Type: {chunk.get('chunk_type', 'N/A')}\n\n"
                f"{chunk.get('text', '')}\n"
            )
            context_parts.append(chunk_text)
        
        # Combine context
        context = "\n\n---\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[:self.max_context_length] + "\n\n[... content truncated ...]"
            logger.warning(f"Context truncated to {self.max_context_length} characters")
        
        return context, source_metadata

    def generate_mcqs(self, topic: str, difficulty: str = "medium", 
                     chapter: Optional[int] = None) -> RAGPipelineResult:
        """
        Generate MCQs for a given topic with JSON output
        
        Args:
            topic: The topic/query for MCQ generation
            difficulty: Difficulty level (easy, medium, hard)
            chapter: Optional chapter filter
        
        Returns:
            RAGPipelineResult with MCQs and metadata
        """
        import time
        total_start = time.time()
        
        logger.info(f"Starting MCQ generation for topic: {topic} (difficulty: {difficulty})")
        
        try:
            # Step 1: Retrieve relevant chunks
            retrieval_start = time.time()
            filters = {'chapter_number': chapter} if chapter else None
            chunks = self.retrieve_chunks(topic, filters)
            retrieval_time = time.time() - retrieval_start
            
            if not chunks:
                error_msg = f"No relevant content found for topic: {topic}"
                logger.error(error_msg)
                return RAGPipelineResult(
                    success=False,
                    topic=topic,
                    mcqs=[],
                    retrieval_time=retrieval_time,
                    generation_time=0,
                    total_time=time.time() - total_start,
                    error_message=error_msg
                )
            
            # Step 2: Format context
            context, source_metadata = self.format_context(chunks)
            
            # Step 3: Generate MCQs with JSON output
            generation_start = time.time()
            mcqs = self.mcq_generator.generate_mcqs(context, topic, difficulty)
            generation_time = time.time() - generation_start
            
            # Attach source metadata to each MCQ
            for mcq in mcqs:
                mcq.source_chunks = source_metadata
            
            total_time = time.time() - total_start
            
            # Get database statistics
            db_stats = self.db.get_statistics()
            
            result = RAGPipelineResult(
                success=True,
                topic=topic,
                mcqs=mcqs,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                total_time=total_time,
                database_stats=db_stats
            )
            
            logger.info(
                f"MCQ generation completed successfully. "
                f"Generated {len(mcqs)} questions in {total_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in MCQ generation pipeline: {e}", exc_info=True)
            return RAGPipelineResult(
                success=False,
                topic=topic,
                mcqs=[],
                retrieval_time=0,
                generation_time=0,
                total_time=time.time() - total_start,
                error_message=str(e)
            )

    def close(self):
        """Close database connection"""
        self.db.close()
        logger.info("RAG Pipeline JSON closed")

# =============================================================================
# === CLI AND MAIN ===
# =============================================================================

def main():
    """Main entry point for testing JSON MCQ generation"""
    parser = argparse.ArgumentParser(
        description='RAG Pipeline for Biology MCQ Generation with JSON Output',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--topic', required=True, help='Topic for MCQ generation')
    parser.add_argument('--difficulty', default='medium', choices=['easy', 'medium', 'hard'],
                       help='Difficulty level for questions')
    parser.add_argument('--chapter', type=int, help='Chapter number filter')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    try:
        pipeline = RAGPipelineJSON(args.config)
        
        # Generate MCQs
        result = pipeline.generate_mcqs(args.topic, args.difficulty, args.chapter)
        
        if result.success:
            print(f"\n=== MCQ GENERATION RESULTS ===")
            print(f"Topic: {result.topic}")
            print(f"Questions Generated: {len(result.mcqs)}")
            print(f"Retrieval Time: {result.retrieval_time:.2f}s")
            print(f"Generation Time: {result.generation_time:.2f}s")
            print(f"Total Time: {result.total_time:.2f}s")
            print("="*50)
            
            for i, mcq in enumerate(result.mcqs, 1):
                print(f"\nQ{i}. {mcq.question_text}")
                print(f"    Difficulty: {mcq.difficulty.upper()}")
                
                for option in mcq.options:
                    print(f"    {option.letter}) {option.text}")
                
                print(f"\n    ✓ Correct Answer: {mcq.correct_answer}")
                print(f"    Explanation: {mcq.explanation}")
            
            # Save to file if requested
            if args.output:
                output_data = {
                    'topic': result.topic,
                    'success': result.success,
                    'mcqs': [mcq.to_dict() for mcq in result.mcqs],
                    'timing': {
                        'retrieval_time': result.retrieval_time,
                        'generation_time': result.generation_time,
                        'total_time': result.total_time
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Results saved to {args.output}")
                print(f"\nResults saved to: {args.output}")
        else:
            print(f"\n❌ MCQ generation failed: {result.error_message}")
        
        pipeline.close()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
