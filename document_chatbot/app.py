# Document Research & Theme Identification Chatbot
# Complete minimal implementation with all core features

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st
import PyPDF2
import pytesseract
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
from datetime import datetime
import pandas as pd
import re

# Configuration
class Config:
    """Configuration settings for the chatbot"""
    DB_PATH = "documents.db"
    UPLOAD_DIR = "uploads"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
    MODEL_NAME = "all-MiniLM-L6-v2"  # Sentence transformer model
    CHUNK_SIZE = 500  # Text chunk size for processing
    
# Initialize OpenAI
openai.api_key = Config.OPENAI_API_KEY

class DocumentProcessor:
    """Handles document upload, OCR, and text extraction"""
    
    def __init__(self):
        self.upload_dir = Path(Config.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF files"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    text += f"[Page {page_num + 1}] {page_text}\n"
                return text
        except Exception as e:
            st.error(f"Error extracting PDF: {str(e)}")
            return ""
    
    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from images using OCR"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return f"[OCR Extracted] {text}"
        except Exception as e:
            st.error(f"Error with OCR: {str(e)}")
            return ""
    
    def process_document(self, uploaded_file) -> Dict[str, Any]:
        """Process uploaded document and extract text"""
        file_path = self.upload_dir / uploaded_file.name
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            text = self.extract_text_from_pdf(str(file_path))
        elif uploaded_file.type.startswith("image/"):
            text = self.extract_text_from_image(str(file_path))
        else:
            text = "Unsupported file format"
        
        return {
            "filename": uploaded_file.name,
            "filepath": str(file_path),
            "text": text,
            "file_type": uploaded_file.type,
            "upload_date": datetime.now().isoformat()
        }

class DatabaseManager:
    """Handles document storage and retrieval"""
    
    def __init__(self):
        self.db_path = Config.DB_PATH
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                text TEXT NOT NULL,
                file_type TEXT,
                upload_date TEXT,
                processed_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_text TEXT,
                chunk_index INTEGER,
                embedding BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_document(self, doc_data: Dict[str, Any]) -> int:
        """Save document to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO documents (filename, filepath, text, file_type, upload_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            doc_data['filename'],
            doc_data['filepath'],
            doc_data['text'],
            doc_data['file_type'],
            doc_data['upload_date']
        ))
        
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doc_id
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM documents')
        documents = []
        for row in cursor.fetchall():
            documents.append({
                'id': row[0],
                'filename': row[1],
                'filepath': row[2],
                'text': row[3],
                'file_type': row[4],
                'upload_date': row[5],
                'processed_date': row[6]
            })
        
        conn.close()
        return documents
    
    def save_text_chunks(self, doc_id: int, chunks: List[str], embeddings: np.ndarray):
        """Save text chunks and embeddings"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cursor.execute('''
                INSERT INTO text_chunks (doc_id, chunk_text, chunk_index, embedding)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, chunk, i, embedding.tobytes()))
        
        conn.commit()
        conn.close()

class VectorSearchEngine:
    """Handles semantic search using sentence transformers and FAISS"""
    
    def __init__(self):
        self.model = SentenceTransformer(Config.MODEL_NAME)
        self.index = None
        self.chunk_metadata = []
    
    def create_chunks(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        # Simple chunking by sentences with overlap
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < Config.CHUNK_SIZE:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index from documents"""
        all_chunks = []
        self.chunk_metadata = []
        
        for doc in documents:
            chunks = self.create_chunks(doc['text'])
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                self.chunk_metadata.append({
                    'doc_id': doc['id'],
                    'filename': doc['filename'],
                    'chunk_index': i,
                    'chunk_text': chunk
                })
        
        if all_chunks:
            # Generate embeddings
            embeddings = self.model.encode(all_chunks)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if not self.index:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunk_metadata):
                result = self.chunk_metadata[idx].copy()
                result['score'] = float(score)
                results.append(result)
        
        return results

class ThemeAnalyzer:
    """Analyzes documents and identifies themes using OpenAI"""
    
    def __init__(self):
        pass
    
    def analyze_query_responses(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze search results and identify themes"""
        # Group results by document
        doc_responses = {}
        for result in search_results:
            doc_id = result['doc_id']
            filename = result['filename']
            
            if doc_id not in doc_responses:
                doc_responses[doc_id] = {
                    'filename': filename,
                    'chunks': [],
                    'doc_id': f"DOC{doc_id:03d}"
                }
            
            doc_responses[doc_id]['chunks'].append({
                'text': result['chunk_text'],
                'score': result['score']
            })
        
        # Generate individual document responses
        individual_responses = []
        for doc_id, data in doc_responses.items():
            # Combine top chunks for this document
            combined_text = " ".join([chunk['text'] for chunk in data['chunks'][:3]])
            
            # Generate response for this document
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Extract the most relevant information to answer the user's query. Be concise and factual."},
                        {"role": "user", "content": f"Query: {query}\n\nDocument content: {combined_text}\n\nProvide a direct answer with key information."}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                
                answer = response.choices[0].message.content
                
                individual_responses.append({
                    'doc_id': data['doc_id'],
                    'filename': data['filename'],
                    'answer': answer,
                    'citation': f"Document {data['doc_id']}"
                })
            except Exception as e:
                individual_responses.append({
                    'doc_id': data['doc_id'],
                    'filename': data['filename'],
                    'answer': f"Error processing: {str(e)}",
                    'citation': f"Document {data['doc_id']}"
                })
        
        # Identify themes across documents
        themes = self.identify_themes(query, individual_responses)
        
        return {
            'individual_responses': individual_responses,
            'themes': themes,
            'query': query
        }
    
    def identify_themes(self, query: str, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify common themes across document responses"""
        # Combine all responses for theme analysis
        all_responses = "\n".join([f"DOC{i+1}: {resp['answer']}" for i, resp in enumerate(responses)])
        
        try:
            theme_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Identify 2-4 common themes from the document responses. For each theme, provide a title and list the supporting document IDs."},
                    {"role": "user", "content": f"Query: {query}\n\nDocument Responses:\n{all_responses}\n\nIdentify common themes and supporting documents."}
                ],
                max_tokens=400,
                temperature=0.5
            )
            
            theme_text = theme_response.choices[0].message.content
            
            # Parse themes (simple parsing - in production, use structured output)
            themes = []
            theme_lines = theme_text.split('\n')
            current_theme = None
            
            for line in theme_lines:
                if line.strip() and ('Theme' in line or 'theme' in line):
                    if current_theme:
                        themes.append(current_theme)
                    current_theme = {
                        'title': line.strip(),
                        'description': '',
                        'supporting_docs': []
                    }
                elif current_theme and line.strip():
                    if 'DOC' in line.upper():
                        # Extract document IDs
                        doc_matches = re.findall(r'DOC\d+', line.upper())
                        current_theme['supporting_docs'].extend(doc_matches)
                    else:
                        current_theme['description'] += line.strip() + ' '
            
            if current_theme:
                themes.append(current_theme)
            
            return themes
            
        except Exception as e:
            return [{
                'title': 'Analysis Error',
                'description': f'Error identifying themes: {str(e)}',
                'supporting_docs': []
            }]

# Streamlit App
class ChatbotApp:
    """Main Streamlit application"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.db_manager = DatabaseManager()
        self.search_engine = VectorSearchEngine()
        self.theme_analyzer = ThemeAnalyzer()
        
        # Initialize session state
        if 'documents_loaded' not in st.session_state:
            st.session_state.documents_loaded = False
        if 'search_index_built' not in st.session_state:
            st.session_state.search_index_built = False
    
    def run(self):
        """Main application runner"""
        st.set_page_config(
            page_title="Document Research & Theme Identification Chatbot",
            page_icon="📚",
            layout="wide"
        )
        
        st.title("📚 Document Research & Theme Identification Chatbot")
        st.markdown("Upload documents, ask questions, and discover themes across your document collection.")
        
        # Sidebar for document management
        with st.sidebar:
            st.header("📁 Document Management")
            self.document_upload_section()
            self.document_list_section()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🔍 Query Interface")
            self.query_interface()
        
        with col2:
            st.header("📊 Document Statistics")
            self.document_stats()
    
    def document_upload_section(self):
        """Document upload interface"""
        st.subheader("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or Image files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload at least 75 documents for comprehensive analysis"
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    # Process document
                    doc_data = self.doc_processor.process_document(file)
                    
                    # Save to database
                    self.db_manager.save_document(doc_data)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ All documents processed successfully!")
                st.session_state.documents_loaded = True
                st.session_state.search_index_built = False  # Need to rebuild index
                
                # Rebuild search index
                self.rebuild_search_index()
    
    def document_list_section(self):
        """Display uploaded documents"""
        documents = self.db_manager.get_all_documents()
        
        if documents:
            st.subheader(f"📋 Uploaded Documents ({len(documents)})")
            
            for doc in documents[:10]:  # Show first 10
                with st.expander(f"📄 {doc['filename']}"):
                    st.write(f"**Type:** {doc['file_type']}")
                    st.write(f"**Uploaded:** {doc['upload_date']}")
                    st.write(f"**Text Preview:** {doc['text'][:200]}...")
            
            if len(documents) > 10:
                st.write(f"... and {len(documents) - 10} more documents")
        else:
            st.info("No documents uploaded yet.")
    
    def rebuild_search_index(self):
        """Rebuild the search index"""
        with st.spinner("Building search index..."):
            documents = self.db_manager.get_all_documents()
            self.search_engine.build_index(documents)
            st.session_state.search_index_built = True
        st.success("Search index built successfully!")
    
    def query_interface(self):
        """Main query interface"""
        documents = self.db_manager.get_all_documents()
        
        if not documents:
            st.warning("Please upload documents first.")
            return
        
        if not st.session_state.search_index_built:
            if st.button("🔨 Build Search Index"):
                self.rebuild_search_index()
            return
        
        # Query input
        query = st.text_input(
            "Enter your research question:",
            placeholder="What are the main regulatory compliance issues discussed?",
            help="Ask questions about themes, patterns, or specific information across your documents"
        )
        
        if query and st.button("🔍 Search & Analyze"):
            with st.spinner("Searching documents and identifying themes..."):
                # Search for relevant content
                search_results = self.search_engine.search(query, top_k=15)
                
                if search_results:
                    # Analyze and identify themes
                    analysis_results = self.theme_analyzer.analyze_query_responses(query, search_results)
                    
                    # Display results
                    self.display_results(analysis_results)
                else:
                    st.warning("No relevant content found for your query.")
    
    def display_results(self, results: Dict[str, Any]):
        """Display analysis results"""
        st.subheader("📋 Individual Document Responses")
        
        # Create DataFrame for individual responses
        df_data = []
        for resp in results['individual_responses']:
            df_data.append({
                'Document ID': resp['doc_id'],
                'Filename': resp['filename'],
                'Extracted Answer': resp['answer'],
                'Citation': resp['citation']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Display themes
        st.subheader("🎯 Identified Themes")
        
        if results['themes']:
            for i, theme in enumerate(results['themes'], 1):
                with st.expander(f"Theme {i}: {theme['title']}", expanded=True):
                    st.write(f"**Description:** {theme['description']}")
                    if theme['supporting_docs']:
                        st.write(f"**Supporting Documents:** {', '.join(theme['supporting_docs'])}")
                    else:
                        st.write("**Supporting Documents:** All analyzed documents")
        else:
            st.info("No clear themes identified in the current analysis.")
        
        # Download results
        st.subheader("💾 Export Results")
        
        # Create downloadable report
        report = self.create_report(results)
        st.download_button(
            label="📥 Download Analysis Report",
            data=report,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    def create_report(self, results: Dict[str, Any]) -> str:
        """Create downloadable analysis report"""
        report = f"""
DOCUMENT RESEARCH & THEME ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Query: {results['query']}

=====================================
INDIVIDUAL DOCUMENT RESPONSES
=====================================

"""
        for resp in results['individual_responses']:
            report += f"""
Document ID: {resp['doc_id']}
Filename: {resp['filename']}
Answer: {resp['answer']}
Citation: {resp['citation']}
---

"""
        
        report += """
=====================================
IDENTIFIED THEMES
=====================================

"""
        
        for i, theme in enumerate(results['themes'], 1):
            report += f"""
Theme {i}: {theme['title']}
Description: {theme['description']}
Supporting Documents: {', '.join(theme['supporting_docs']) if theme['supporting_docs'] else 'All analyzed documents'}
---

"""
        
        return report
    
    def document_stats(self):
        """Display document statistics"""
        documents = self.db_manager.get_all_documents()
        
        if documents:
            st.metric("Total Documents", len(documents))
            
            # File type distribution
            file_types = {}
            total_text_length = 0
            
            for doc in documents:
                file_type = doc['file_type']
                file_types[file_type] = file_types.get(file_type, 0) + 1
                total_text_length += len(doc['text'])
            
            st.subheader("📊 File Types")
            for file_type, count in file_types.items():
                st.write(f"• {file_type}: {count}")
            
            st.metric("Total Text Length", f"{total_text_length:,} characters")
            st.metric("Average Document Length", f"{total_text_length // len(documents):,} characters")
            
            # Status indicators
            st.subheader("🔧 System Status")
            st.write(f"✅ Documents Loaded: {st.session_state.documents_loaded}")
            st.write(f"✅ Search Index Built: {st.session_state.search_index_built}")
        else:
            st.metric("Total Documents", 0)
            st.info("Upload documents to see statistics.")

# Main execution
if __name__ == "__main__":
    # Create required directories
    os.makedirs(Config.UPLOAD_DIR, exist_ok=True)
    
    # Run the app
    app = ChatbotApp()
    app.run()