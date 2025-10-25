# #!/usr/bin/env python3
# """
# Quick fix: Load chunks into database
# """

# import json
# from pathlib import Path
# from pgvector_database import PGVectorDatabase, DocumentChunk
# from datetime import datetime
# import re
# import ast

# # Configuration - UPDATE THESE TO MATCH YOUR config.json
# DB_CONFIG = {
#     'host': 'localhost',
#     'port': 5432,
#     'database': 'vectorDatabase',
#     'user': 'rag_user',  # Changed from 'postgres' to 'rag_user'
#     'password': 'mypassword123'
# }

# CHUNKS_DIR = Path('chunks_hierarchical')

# print("\n" + "="*70)
# print("LOADING CHUNKS INTO DATABASE")
# print("="*70)

# # Initialize database connection
# print("\n1. Connecting to database...")
# try:
#     db = PGVectorDatabase(**DB_CONFIG)
#     print("   ✓ Connected successfully")
# except Exception as e:
#     print(f"   ✗ Connection failed: {e}")
#     print("\n   Check your config.json database credentials:")
#     print(f"     user: {DB_CONFIG['user']}")
#     print(f"     password: {DB_CONFIG['password']}")
#     exit(1)

# # Find all chunk files
# print(f"\n2. Finding chunk files in {CHUNKS_DIR}...")
# chunk_files = list(CHUNKS_DIR.glob("*.txt"))
# print(f"   ✓ Found {len(chunk_files)} chunk files")

# if len(chunk_files) == 0:
#     print("   ✗ No chunk files found!")
#     db.close()
#     exit(1)

# # Load chunks
# print(f"\n3. Loading chunks into database...")
# print("   This may take a minute...\n")

# loaded_count = 0
# failed_count = 0

# for i, filepath in enumerate(chunk_files):
#     try:
#         # Parse chunk file
#         with open(filepath, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         parts = content.split('-------------------- CONTENT --------------------')
#         if len(parts) != 2:
#             print(f"   ✗ {filepath.name}: Invalid format")
#             failed_count += 1
#             continue
        
#         metadata_text = parts[0]
#         text_content = parts[1].strip()
        
#         # Extract metadata
#         metadata = {}
#         for line in metadata_text.split('\n'):
#             if ':' in line:
#                 key, value = line.split(':', 1)
#                 key = key.strip()
#                 value = value.strip()
                
#                 if key == 'ID':
#                     metadata['chunk_id'] = value
#                 elif key == 'Chapter':
#                     try:
#                         metadata['chapter_number'] = int(value)
#                     except:
#                         metadata['chapter_number'] = 1
#                 elif key == 'Hierarchy':
#                     try:
#                         hierarchy = ast.literal_eval(value)
#                         metadata['section_title'] = hierarchy.get('h2', hierarchy.get('h1', 'Unknown'))
#                     except:
#                         metadata['section_title'] = 'Unknown'
#                 elif key == 'Source File':
#                     metadata['source_file'] = value
#                 elif key == 'Pages':
#                     try:
#                         metadata['page_numbers'] = ast.literal_eval(value)
#                     except:
#                         metadata['page_numbers'] = []
        
#         # Set defaults
#         metadata.setdefault('chunk_id', filepath.stem)
#         metadata.setdefault('chapter_number', 1)
#         metadata.setdefault('section_title', 'Unknown Section')
#         metadata.setdefault('source_file', filepath.name)
#         metadata.setdefault('page_numbers', [])
        
#         # Generate embedding
#         embedding = db.generate_embedding(text_content)
        
#         # Create and insert chunk
#         chunk = DocumentChunk(
#             chunk_id=metadata['chunk_id'],
#             text=text_content,
#             embedding=embedding,
#             chapter_number=metadata['chapter_number'],
#             section_title=metadata['section_title'],
#             page_numbers=metadata['page_numbers'],
#             word_count=len(text_content.split()),
#             source_file=metadata['source_file'],
#             chunk_type='paragraph',
#             textbook_name='Class X Biology',
#             created_at=datetime.now().isoformat()
#         )
        
#         db.insert_chunk(chunk)
#         loaded_count += 1
        
#         if (i + 1) % 10 == 0:
#             print(f"   Loaded {i+1}/{len(chunk_files)} chunks...")
    
#     except Exception as e:
#         print(f"   ✗ Error with {filepath.name}: {str(e)[:50]}")
#         failed_count += 1
#         continue

# print(f"\n4. Results:")
# print(f"   ✓ Successfully loaded: {loaded_count} chunks")
# if failed_count > 0:
#     print(f"   ✗ Failed: {failed_count} chunks")

# # Verify
# print(f"\n5. Verifying...")
# stats = db.get_statistics()
# print(f"   Total chunks in database: {stats['total_chunks']}")
# print(f"   Total words: {stats['total_words']}")
# print(f"   Chunks by chapter: {stats['chunks_by_chapter']}")

# db.close()

# print("\n" + "="*70)
# print("✓ DONE! Now you can run MCQ generation.")
# print("="*70)
# print("\nTest it with:")
# print("  python rag_pipeline.py --stats")
# print("  python rag_pipeline.py --topic 'photosynthesis'")
# print("  python rag_pipeline.py --interactive")

#!/usr/bin/env python3
"""
Fixed: Load chunks with better section title extraction


"""

import json
from pathlib import Path
from pgvector_database import PGVectorDatabase, DocumentChunk
from datetime import datetime
import re
import ast

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'vectorDatabase',
    'user': 'rag_user',
    'password': 'mypassword123'
}

CHUNKS_DIR = Path('chunks_hierarchical')

def extract_section_title(hierarchy: dict, text_snippet: str) -> str:
    """
    Extract a meaningful section title from hierarchy or text
    """
    if not hierarchy:
        return 'Unknown'
    
    # Try h3 first (most specific)
    if 'h3' in hierarchy and hierarchy['h3']:
        title = hierarchy['h3']
        # Remove numbering like "1.", "A)", "i."
        title = re.sub(r'^[\d\w][.)]\s*', '', title)
        if title.strip():
            return title.strip()
    
    # Try h2
    if 'h2' in hierarchy and hierarchy['h2']:
        title = hierarchy['h2']
        # Remove numbering and excessive whitespace
        title = re.sub(r'^[\d\w][.)]\s*', '', title)
        # Take first meaningful part
        if '(' in title:
            title = title.split('(')[0]
        if title.strip():
            return title.strip()[:100]  # Cap at 100 chars
    
    # Try h1
    if 'h1' in hierarchy and hierarchy['h1']:
        title = hierarchy['h1']
        if title.strip():
            return title.strip()
    
    # Fallback: use first 60 chars of text
    if text_snippet:
        text = text_snippet.strip()
        # Remove newlines and extra spaces
        text = ' '.join(text.split())
        return text[:60] + ('...' if len(text) > 60 else '')
    
    return 'Unknown'

print("\n" + "="*70)
print("LOADING CHUNKS INTO DATABASE (FIXED)")
print("="*70)

# Initialize database
print("\n1. Connecting to database...")
try:
    db = PGVectorDatabase(**DB_CONFIG)
    print("   ✓ Connected successfully")
except Exception as e:
    print(f"   ✗ Connection failed: {e}")
    exit(1)

# Find chunks
print(f"\n2. Finding chunk files in {CHUNKS_DIR}...")
chunk_files = list(CHUNKS_DIR.glob("*.txt"))
print(f"   ✓ Found {len(chunk_files)} chunk files")

if len(chunk_files) == 0:
    print("   ✗ No chunk files found!")
    db.close()
    exit(1)

# Load chunks
print(f"\n3. Loading chunks into database...")
print("   Processing...\n")

loaded_count = 0
failed_count = 0

for i, filepath in enumerate(chunk_files):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        parts = content.split('-------------------- CONTENT --------------------')
        if len(parts) != 2:
            failed_count += 1
            continue
        
        metadata_text = parts[0]
        text_content = parts[1].strip()
        
        # Extract metadata
        metadata = {}
        for line in metadata_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'ID':
                    metadata['chunk_id'] = value
                elif key == 'Chapter':
                    try:
                        metadata['chapter_number'] = int(value)
                    except:
                        metadata['chapter_number'] = 1
                elif key == 'Hierarchy':
                    try:
                        metadata['hierarchy'] = ast.literal_eval(value)
                    except:
                        metadata['hierarchy'] = {}
                elif key == 'Source File':
                    metadata['source_file'] = value
                elif key == 'Pages':
                    try:
                        metadata['page_numbers'] = ast.literal_eval(value)
                    except:
                        metadata['page_numbers'] = []
        
        # Set defaults
        metadata.setdefault('chunk_id', filepath.stem)
        metadata.setdefault('chapter_number', 1)
        metadata.setdefault('source_file', filepath.name)
        metadata.setdefault('page_numbers', [])
        metadata.setdefault('hierarchy', {})
        
        # Extract section title with improved logic
        hierarchy = metadata.get('hierarchy', {})
        section_title = extract_section_title(hierarchy, text_content)
        
        # Generate embedding
        embedding = db.generate_embedding(text_content)
        
        # Create and insert chunk
        chunk = DocumentChunk(
            chunk_id=metadata['chunk_id'],
            text=text_content,
            embedding=embedding,
            chapter_number=metadata['chapter_number'],
            section_title=section_title,
            page_numbers=metadata['page_numbers'],
            word_count=len(text_content.split()),
            source_file=metadata['source_file'],
            chunk_type='paragraph',
            textbook_name='Class X Biology',
            created_at=datetime.now().isoformat()
        )
        
        db.insert_chunk(chunk)
        loaded_count += 1
        
        if (i + 1) % 20 == 0:
            print(f"   Loaded {i+1}/{len(chunk_files)} chunks...")
    
    except Exception as e:
        failed_count += 1
        continue

print(f"\n4. Results:")
print(f"   ✓ Successfully loaded: {loaded_count} chunks")
if failed_count > 0:
    print(f"   ✗ Failed: {failed_count} chunks")

# Verify with sample
print(f"\n5. Sample section titles from database:")
try:
    conn = db.connection
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chapter_number, section_title, COUNT(*) 
            FROM document_chunks 
            GROUP BY chapter_number, section_title 
            LIMIT 10
        """)
        for row in cur.fetchall():
            print(f"   Chapter {row[0]}: {row[1]} ({row[2]} chunks)")
except:
    pass

stats = db.get_statistics()
print(f"\n   Total chunks in database: {stats['total_chunks']}")

db.close()

print("\n" + "="*70)
print("✓ DONE! Section titles should now be meaningful.")
print("="*70)
print("\nTest with:")
print("  python rag_pipeline.py --topic 'Punnett square'")