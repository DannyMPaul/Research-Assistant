# Research Assistant

A sophisticated, locally-run AI-powered research assistant to help you analyze and chat with your documents.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is a web-based application that allows you to upload documents (PDFs, DOCX, and TXT files) and interact with them through a chat interface. It's designed to be a powerful tool for researchers, students, and anyone who needs to quickly find information within a collection of documents.

Unlike other document chat applications that rely on cloud services, this Research Assistant runs entirely on your local machine, ensuring your data remains private and secure. It uses a combination of traditional keyword search, modern semantic search, and a powerful local large language model (LLM) to provide accurate and context-aware answers to your questions.

## Features

- **Local First**: Everything runs on your machine. No data is sent to the cloud.
- **Multiple Search Methods**:
  - **Keyword Search**: The classic way to find exact words or phrases.
  - **Semantic Search**: Finds relevant passages based on meaning, not just keywords.
  - **AI-Enhanced Search**: Uses a large language model to understand your questions and find the best answers.
- **Document Management**:
  - Upload and manage your documents.
  - View document details and previews.
- **Conversational Interface**:
  - Chat with your documents in natural language.
  - Conversations are saved automatically.
- **Advanced Indexing**:
  - Creates a hierarchical index of your documents for faster and more accurate retrieval.
- **User-Friendly Interface**:
  - Dark mode and a clean, modern design.
  - Keyboard shortcuts for power users.

## How It Works

The application is built with a Python backend and a vanilla JavaScript frontend.

1.  **Document Upload**: When you upload a document, the backend extracts the text and splits it into smaller chunks.
2.  **Indexing**: These chunks are then indexed in two ways:
    - A traditional keyword index.
    - A vector index using sentence-transformers and FAISS for semantic search.
3.  **Chat**: When you ask a question, the application uses a multi-step process to find the answer:
    - It first tries to find relevant chunks using semantic search.
    - If that fails, it falls back to keyword search.
    - The most relevant chunks are then passed to a local large language model (Llama-3.2-1B-Instruct) along with your question.
    - The LLM generates an answer based on the provided context.

## Technology Stack

### Backend

- **Framework**: FastAPI
- **AI/ML**:
  - **LLM**: Llama-3.2-1B-Instruct
  - **Embeddings**: `sentence-transformers`
  - **Vector Store**: FAISS
- **Document Processing**: PyMuPDF, python-docx

### Frontend

- **Framework**: Vanilla JavaScript
- **Styling**: Tailwind CSS

## Getting Started

### Prerequisites

- Python 3.11+
- 4GB+ RAM (8GB+ recommended)
- A GPU with CUDA support is recommended for the best performance, but the application will fall back to CPU if a GPU is not available.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/research-assistant.git
    cd research-assistant
    ```
2.  Install the backend dependencies:
    ```bash
    cd backend
    pip install -r requirements.txt
    ```

### Configuration

You can configure the application by creating a `.env` file in the `backend` directory. See the `.env.example` file for a list of available options.

### Running the Application

1.  Start the backend server:
    ```bash
    python backend/app.py
    ```
2.  Open your browser and navigate to `http://localhost:8000`.

## Usage

Once the application is running, you can start uploading documents and asking questions. The interface is designed to be intuitive, but here are a few tips:

- Use the tabs at the top to switch between the document manager, search, and chat.
- Use the search bar to quickly find documents or start a new chat.
- Use the keyboard shortcuts (`?` for help) to speed up your workflow.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any ideas or suggestions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
