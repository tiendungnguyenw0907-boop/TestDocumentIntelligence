# TestDocumentIntelligence

A comprehensive document intelligence and processing system for extracting, understanding, and leveraging information from documents using modern AI techniques.

## 📋 Overview

TestDocumentIntelligence is a Python-based project designed to handle the complete lifecycle of document processing, from ingestion to intelligent retrieval and understanding. It combines multiple AI capabilities including document extraction, indexing, and retrieval-augmented generation (RAG).

## 📁 Project Structure

```
TestDocumentIntelligence/
├── context/              # Context management and storage
├── extraction/           # Document content extraction
├── indexing/             # Indexing strategies and operations
├── ingestion/            # Document ingestion pipeline
├── knowledge/            # Knowledge base management
├── page_understanding/   # Page-level content understanding
├── pipelines/            # End-to-end processing pipelines
├── rag/                  # Retrieval-Augmented Generation
├── schemas/              # Data schemas and models
├── structure/            # Document structure analysis
└── table/                # Table extraction and processing
```

## 🔧 Key Modules

### **Ingestion** (`ingestion/`)
Handles the initial document upload and ingestion process. Manages data validation and initial preprocessing.

### **Extraction** (`extraction/`)
Extracts structured content from documents including text, metadata, and key information.

### **Structure** (`structure/`)
Analyzes and understands the hierarchical structure of documents, including sections, headings, and layout.

### **Page Understanding** (`page_understanding/`)
Provides deep understanding of individual page content, layout analysis, and semantic comprehension.

### **Table** (`table/`)
Specialized module for extracting, parsing, and processing tabular data from documents.

### **Schemas** (`schemas/`)
Defines data models and schemas for consistent data representation throughout the system.

### **Indexing** (`indexing/`)
Manages document indexing for efficient search and retrieval operations.

### **Knowledge** (`knowledge/`)
Manages the knowledge base and stores extracted information for intelligent querying.

### **Context** (`context/`)
Maintains contextual information across processing operations and user interactions.

### **RAG** (`rag/`)
Implements Retrieval-Augmented Generation for intelligent document question-answering and information synthesis.

### **Pipelines** (`pipelines/`)
Orchestrates the complete processing workflow, connecting all modules into end-to-end pipelines.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Dependencies (see requirements file)

### Installation

```bash
# Clone the repository
git clone https://github.com/tiendungnguyenw0907-boop/TestDocumentIntelligence.git

# Navigate to the project directory
cd TestDocumentIntelligence

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
# Example: Document ingestion and processing
from pipelines import DocumentProcessingPipeline

pipeline = DocumentProcessingPipeline()
results = pipeline.process_document("path/to/document.pdf")
```

## 💡 Features

- **Document Ingestion**: Easy integration of various document formats
- **Content Extraction**: Intelligent extraction of text, tables, and structured data
- **Document Understanding**: Deep analysis of document structure and semantics
- **Intelligent Search**: RAG-based retrieval and question-answering
- **Scalable Architecture**: Modular design for easy extension and customization
- **Knowledge Management**: Efficient storage and retrieval of document knowledge

## 🔄 Processing Pipeline

1. **Ingestion** → Load and validate documents
2. **Structure Analysis** → Understand document layout and organization
3. **Page Understanding** → Extract page-level insights
4. **Extraction** → Parse content into structured formats
5. **Table Processing** → Handle tabular data specially
6. **Indexing** → Create searchable indexes
7. **Knowledge Storage** → Store for intelligent retrieval
8. **RAG** → Enable question-answering and synthesis

## 📝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## 📄 License

This project is open source and available under the appropriate license.

## 👤 Author

**tiendungnguyenw0907-boop**

## 🌐 Repository

[GitHub Repository](https://github.com/tiendungnguyenw0907-boop/TestDocumentIntelligence)

---

For more information or questions about this project, please open an issue on GitHub.
