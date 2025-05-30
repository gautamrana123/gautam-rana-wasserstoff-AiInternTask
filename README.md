#Document Research & Theme Identification Chatbot
A complete implementation of an AI-powered chatbot that processes 75+ documents, performs semantic search, and identifies common themes across documents with detailed citations.
🚀 Quick Start Guide
Prerequisites

Python 3.8 or higher
VS Code (recommended)
OpenAI API key (for theme analysis)
Tesseract OCR (for image processing)

1. Setup in VS Code
Step 1: Create Project Directory
bashmkdir document_chatbot
cd document_chatbot
Step 2: Create Virtual Environment
bash# Create virtual environment
python -m venv chatbot_env

# Activate virtual environment
# Windows:
chatbot_env\Scripts\activate
# Mac/Linux:
source chatbot_env/bin/activate
Step 3: Install Dependencies
Create requirements.txt with the provided content, then:
bashpip install -r requirements.txt
Step 4: Install Tesseract OCR

Windows: Download from https://github.com/tesseract-ocr/tesseract/wiki
Mac: brew install tesseract
Linux: sudo apt-get install tesseract-ocr

Step 5: Setup OpenAI API Key

Get API key from https://platform.openai.com/api-keys
Create .env file in project root:

OPENAI_API_KEY=your-actual-api-key-here
Step 6: Create Main Application File
Save the main code as app.py in your project directory.
2. Running the Application
In VS Code Terminal:
bash# Make sure virtual environment is activated
streamlit run app.py
Alternative Method:
bashpython -m streamlit run app.py
The application will open in your browser at http://localhost:8501
📁 Project Structure
document_chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (API keys)
├── .gitignore           # Git ignore file
├── documents.db         # SQLite database (created automatically)
├── uploads/             # Document upload directory (created automatically)
└── chatbot_env/         # Virtual environment directory
🔧 Features Implemented
Core Features ✅

Document Upload: Support for PDF and image files (75+ documents)
OCR Processing: Automatic text extraction from scanned images
Semantic Search: Vector-based search using sentence transformers
Theme Identification: AI-powered theme analysis across documents
Citation Management: Document-level citations with precise references
Database Storage: SQLite database for document persistence

User Interface ✅

Clean Web Interface: Streamlit-based responsive design
Document Management: Upload, view, and manage document collection
Query Interface: Natural language query processing
Results Display: Tabular format for individual responses
Theme Visualization: Clear presentation of identified themes
Export Functionality: Download analysis reports

Technical Implementation ✅

Vector Database: FAISS for efficient semantic search
Text Processing: Chunking and embedding generation
AI Integration: OpenAI GPT for theme analysis
Error Handling: Robust exception handling throughout
Scalable Design: Modular architecture for easy extension

💻 Usage Instructions
1. Upload Documents

Use the sidebar to upload PDF or image files
Minimum 75 documents recommended for comprehensive analysis
Documents are automatically processed and stored

2. Build Search Index

Click "Build Search Index" after uploading documents
This creates vector embeddings for semantic search
Only needs to be done once per document collection

3. Query Documents

Enter natural language questions in the query interface
Examples:

"What are the main regulatory compliance issues?"
"Identify common themes related to penalties"
"What legal frameworks are mentioned across documents?"



4. Review Results

Individual Responses: See how each document answers your query
Identified Themes: AI-generated themes with supporting document citations
Export Report: Download complete analysis as text file

🛠️ Troubleshooting
Common Issues and Solutions
1. "No module named 'streamlit'"
bash# Ensure virtual environment is activated
pip install -r requirements.txt
2. Tesseract not found
bash# Windows: Add tesseract to PATH or install via installer
# Mac: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr
3. OpenAI API errors

Verify your API key in .env file
Check API key permissions and billing
Ensure internet connection

4. Memory issues with large documents

Reduce CHUNK_SIZE in config
Process documents in smaller batches
Increase system RAM if possible

Performance Optimization
For Large Document Collections:

Batch Processing: Upload documents in groups of 20-30
Index Rebuilding: Only rebuild when adding new documents
Query Optimization: Use specific, focused queries
Resource Management: Close unused browser tabs

🔄 Development Workflow
Adding New Features:

Backup Database: Copy documents.db before changes
Test with Sample Data: Use small document set first
Version Control: Use git for code management
Error Logging: Add logging for debugging

Customization Options:

Modify Themes: Adjust theme analysis prompts in ThemeAnalyzer
Change Models: Update sentence transformer model in config
UI Customization: Modify Streamlit components
Database Schema: Extend tables for additional metadata

📊 Technical Specifications
Performance Metrics:

Document Processing: ~2-5 seconds per PDF
Index Building: ~1-3 minutes for 100 documents
Query Response: ~5-10 seconds including theme analysis
Memory Usage: ~500MB-1

