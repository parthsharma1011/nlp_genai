# GenAI Applications

Advanced Generative AI implementations using modern LLMs, vector databases, and agent frameworks.

## Features

### RAG System (`rag.py`)
Retrieval Augmented Generation system that combines document search with language generation.

**Capabilities:**
- Document vectorization using HuggingFace embeddings
- FAISS vector database for similarity search
- LangChain agents with tool integration
- Google Gemini 2.0 Flash for response generation

**Usage:**
```python
python rag.py
```

### Chat with Files (`chat_with_files.py`)
Interactive system for querying documents and files using natural language.

### Configuration Management (`utils/config.py`)
Centralized configuration for API keys and external service URLs.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `utils/config.py` and add your API keys:
```python
class Config:
    GEMINI_API_KEY = "your_gemini_api_key_here"
    SHEET_URL = "your_google_sheets_url"
```

### 3. Run Applications
```bash
# Start RAG system
python rag.py

# Start chat with files
python chat_with_files.py
```

## Architecture

### RAG Pipeline
1. **Document Processing** - Convert text to vector embeddings
2. **Vector Storage** - Store in FAISS database for fast retrieval
3. **Query Processing** - Find relevant documents using similarity search
4. **Response Generation** - Use LLM to generate contextual answers

### Agent Framework
- **Tools Integration** - Custom tools for document search
- **Chain of Thought** - Step-by-step reasoning process
- **Guardrails** - Content filtering and safety measures

## Dependencies

**Core Libraries:**
- `langchain` - LLM orchestration framework
- `langchain-community` - Community extensions
- `langchain-google-genai` - Google Gemini integration
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embeddings

**Supporting Libraries:**
- `transformers` - HuggingFace model hub
- `torch` - PyTorch for deep learning
- `numpy` - Numerical computations
- `python-dotenv` - Environment variable management

## Troubleshooting

### Common Issues

**Import Errors:**
- Ensure all dependencies are installed with correct versions
- Check Python version compatibility (3.8+)

**API Key Issues:**
- Verify API keys are correctly set in config.py
- Check API key permissions and quotas

**Memory Issues:**
- Reduce batch size for large documents
- Use CPU version of FAISS for lower memory usage

### Performance Optimization

**For Large Documents:**
- Implement document chunking
- Use hierarchical indexing
- Consider GPU acceleration for embeddings

**For Production:**
- Implement caching mechanisms
- Use async processing for concurrent requests
- Monitor API usage and costs

## Examples

### Basic RAG Query
```python
from rag import search_documents

# Search for relevant information
results = search_documents("What does the company do?")
print(results)
```

### Custom Document Processing
```python
from langchain.docstore.document import Document

# Add your own documents
custom_docs = [
    Document(page_content="Your custom content here"),
    Document(page_content="More content to search through")
]
```

## Next Steps

1. **Extend Document Sources** - Add PDF, CSV, database connectors
2. **Improve Embeddings** - Fine-tune models for domain-specific content
3. **Add Memory** - Implement conversation history
4. **Scale Infrastructure** - Deploy with proper vector databases
5. **Add Evaluation** - Implement RAG evaluation metrics