#!/usr/bin/env python3
"""
OCR Processor for PDF documents using Dolphin
Processes individual PDF files and returns markdown content
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles OCR processing of PDF documents using Dolphin"""
    
    def __init__(self, dolphin_repo_dir: Optional[str] = None, hf_model_dir: Optional[str] = None):
        """
        Initialize OCR processor with Dolphin
        
        Args:
            dolphin_repo_dir: Path to Dolphin repository (if None, uses default)
            hf_model_dir: Path to Hugging Face model directory (if None, uses default)
        """
        # Configure paths
        if dolphin_repo_dir is None:
            dolphin_repo_dir = os.getenv(
                'DOLPHIN_REPO_DIR',
                '/Users/sahithikaruparthi/Desktop/gradeBoostRAG/pre-processing-pipeline/Dolphin'
            )
        
        self.dolphin_repo_dir = Path(dolphin_repo_dir)
        
        if hf_model_dir is None:
            self.hf_model_dir = self.dolphin_repo_dir / "hf_model"
        else:
            self.hf_model_dir = Path(hf_model_dir)
        
        # Validate paths
        if not self.dolphin_repo_dir.exists():
            raise FileNotFoundError(f"Dolphin repo not found at {self.dolphin_repo_dir}")
        if not self.hf_model_dir.exists():
            raise FileNotFoundError(f"Dolphin hf_model not found at {self.hf_model_dir}")
        
        # Add Dolphin to Python path
        if str(self.dolphin_repo_dir) not in sys.path:
            sys.path.insert(0, str(self.dolphin_repo_dir))
        
        # Create temporary work directory
        self.work_dir = Path(tempfile.mkdtemp(prefix="dolphin_work_"))
        
        # Load Dolphin model
        self.model = self._load_dolphin_model()
        logger.info("OCR Processor initialized successfully")
    
    def _load_dolphin_model(self):
        """Load Dolphin HF model"""
        try:
            from demo_page_hf import DOLPHIN
            logger.info(f"Loading Dolphin model from {self.hf_model_dir}")
            model = DOLPHIN(str(self.hf_model_dir))
            logger.info("Dolphin model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading Dolphin model: {e}")
            raise
    
    def process_pdf(self, pdf_path: Path) -> str:
        """
        Process a PDF file and return markdown content
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Markdown content as string
        """
        try:
            logger.info(f"Starting OCR processing for {pdf_path.name}")
            
            # Step 1: Parse PDF with Dolphin
            logger.info("Step 1: Parsing PDF with Dolphin")
            json_path, recognition_results = self._dolphin_parse_pdf(pdf_path)
            
            # Step 2: Convert to Markdown
            logger.info("Step 2: Converting to Markdown")
            markdown_content = self._dolphin_results_to_markdown(recognition_results)
            
            logger.info(f"OCR processing completed for {pdf_path.name}")
            logger.info(f"Generated markdown with {len(markdown_content)} characters")
            
            return markdown_content
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path.name}: {e}")
            raise
    
    def _dolphin_parse_pdf(self, pdf_path: Path) -> Tuple[str, list]:
        """Use Dolphin's HF document pipeline to parse a PDF"""
        try:
            from demo_page_hf import process_document
            
            json_path, recognition_results = process_document(
                document_path=str(pdf_path),
                model=self.model,
                save_dir=str(self.work_dir),
                max_batch_size=16,
            )
            
            logger.info(f"Dolphin parsed {len(recognition_results)} pages")
            return json_path, recognition_results
            
        except Exception as e:
            logger.error(f"Error in Dolphin PDF parsing: {e}")
            raise
    
    def _dolphin_results_to_markdown(self, recognition_results: list) -> str:
        """Convert Dolphin recognition results to markdown"""
        try:
            from utils.markdown_utils import MarkdownConverter
            
            # Flatten results if needed
            if (recognition_results and 
                isinstance(recognition_results[0], dict) and 
                "elements" in recognition_results[0]):
                flat = []
                for page in recognition_results:
                    flat.extend(page.get("elements", []))
                recognition_results = flat
            
            converter = MarkdownConverter()
            md = converter.convert(recognition_results)
            
            return md
            
        except Exception as e:
            logger.error(f"Error converting to markdown: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary work directory"""
        try:
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir)
                logger.info(f"Cleaned up work directory: {self.work_dir}")
        except Exception as e:
            logger.warning(f"Error cleaning up work directory: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()