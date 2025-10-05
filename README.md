# Document Research Assistant

AI-powered document analysis platform with intelligent Q&A capabilities using Large Language Models.

## Version 4.1.0 - Enhanced User Experience

### Features

- **AI Chat**: Ask questions about your documents with contextual answers
- **Smart Search**: Keyword, semantic, and AI-enhanced search modes
- **Document Support**: PDF, DOCX, TXT with automatic processing
- **Query History**: Auto-saved search suggestions
- **Export**: Save conversations as markdown
- **Keyboard Shortcuts**: Full navigation support

## Quick Start

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Access at http://localhost:8002

## Usage

1. **Upload**: Drag documents to "Manage Documents" → "Process for AI"
2. **Search**: Use different search modes in "Search Documents"
3. **Chat**: Ask questions in "AI Chat" tab
4. **Shortcuts**: Ctrl+1/2/3 for tabs, Ctrl+K for search, ? for help

## Tech Stack

- **Backend**: FastAPI, Python 3.11+
- **AI**: Llama-3.2-1B-Instruct, sentence-transformers, FAISS
- **Frontend**: Vanilla JS, Tailwind CSS

## Requirements

- Python 3.11+
- 8GB+ GPU (recommended) or CPU fallback
- 4GB+ RAM

## License

MIT License
