# Document Research Assistant

AI-powered document analysis platform with intelligent Q&A capabilities using Large Language Models.

## Version 4.2.0 - Enhanced Organization & Interaction

### Features

- **AI Chat**: Ask questions about your documents with contextual answers
- **Smart Search**: Keyword, semantic, and AI-enhanced search modes with advanced filters
- **Document Organization**: Category-based document management with visual thumbnails
- **Interactive Messages**: React to messages with emojis and bookmark important conversations
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

Access at http://localhost:8000

## Usage

1. **Upload**: Drag documents to "Manage Documents" → "Process for AI" → Organize with categories
2. **Search**: Use different search modes with filters (file type, size) in "Search Documents"
3. **Chat**: Ask questions → React with emojis → Bookmark important messages
4. **Shortcuts**: Ctrl+1/2/3 for tabs, Ctrl+K for search, ? for help## Tech Stack

- **Backend**: FastAPI, Python 3.11+
- **AI**: Llama-3.2-1B-Instruct, sentence-transformers, FAISS
- **Frontend**: Vanilla JS, Tailwind CSS

## Requirements

- Python 3.11+
- 8GB+ GPU (recommended) or CPU fallback
- 4GB+ RAM

## License

MIT License
