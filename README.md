# Document Research Assistant

A modern web application for uploading documents and extracting insights using AI-powered analysis.

## Version 0.1.0 - Base Infrastructure

This is the foundational version with core document upload and management functionality.

### Features

- Upload PDF, DOCX, and TXT documents
- Document storage and listing
- Clean, responsive web interface
- RESTful API backend

### Tech Stack

- **Backend**: FastAPI, Python 3.11
- **Frontend**: React, Vite, Tailwind CSS
- **Deployment**: Docker & Docker Compose

## Quick Start

### Development Setup

1. **Clone and navigate to project**

```bash
git clone <your-repo>
cd Research_Assistant
```

2. **Start with Docker Compose**

```bash
docker-compose up --build
```

3. **Access the application**

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Manual Setup

#### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

- `GET /` - API status
- `POST /upload` - Upload document
- `GET /documents` - List uploaded documents
- `GET /api/document/{file_id}/text` - Extract text from document

## Project Structure

```
Research_Assistant/
├── backend/                 # FastAPI backend
│   ├── app.py              # Main application
│   ├── routes/             # API route handlers
│   ├── utils/              # Utility functions
│   └── requirements.txt    # Python dependencies
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   └── App.jsx        # Main app component
│   └── package.json       # Node dependencies
└── docker-compose.yml     # Container orchestration
```

## Upcoming Features (Roadmap)

- **v0.2.0**: Text chunking and basic search
- **v0.3.0**: Vector embeddings and similarity search
- **v0.4.0**: AI-powered Q&A with Llama 4 Scout
- **v0.5.0**: Enhanced UI and error handling

## Contributing

This project follows semantic versioning. Each feature increment will be tagged as a new version for easy tracking.

## License

MIT License
