# Navadhāni: Indian History of Science Chatbot

## Overview

Navadhāni is a specialized academic document chat system focused on papers from the Indian Journal of History of Science (IJHS). The application lets users converse with and query a collection of academic papers, with particular emphasis on Indian mathematics and astronomy.

## Data Source

The app allows users to converse with the contents of ijhs-astro-math-docs.tsv, which has columns:

    #                   int64   (index - ignore)
    paper              object   (title)
    author             object   (author)
    url                object   (url)
    size_in_kb        float64   (size_in_kb - ignore)
    cum_size_in_kb    float64   (cum_size_in_kb - ignore)   
    subject            object   (one of the following: 'math', 'astro') 
    category           object   (one of the following: 'indic', 'western', 'other' - ignore)
    pdf                object   (pdf file name - ignore)
    full_pdf_path      object   (full path to pdf file - ignore)
    pdf_exists           bool   (always True -ignore) 
    has_text             bool   (one of raster or text - ignore)
    text               object   (content)

## Core Features

1. Document search and retrieval via TF-IDF vectorization and FAISS similarity search
2. Text chunking for effective context management
3. Integration with Google's Gemini API for natural language understanding
4. Structured JSON responses with references and confidence levels
5. Ability to handle both document-specific queries and collection-level statistics
6. Composite message handling with prompt splitting for multi-part queries
7. Token usage tracking and conversation history management

## Technical Components

### LLM Integration

- Use google.generativeai as genai
- GEMINI_API_KEY is set in a .env file and loaded using dotenv()
- Uses the "gemini-2.0-flash-exp" model
- Configurable system prompts (default, conversational, expert)

### Vector Database

- Use FAISS for vector search capabilities
- TF-IDF vectorizer for document embedding
- Document chunking with overlapping windows for better context

### Query Processing

- Automatic query type detection (document, aggregation, conversational)
- Composite message handling through prompt splitting
- Support for collection-level statistics and aggregation queries
- Message history navigation

## UI Components (Gradio-based)

- Chat interface with message history
- Statistics panel showing token usage
- Example query dropdown for user guidance
- System prompt selector
- New conversation button
- Message navigation buttons (up/down arrows)
- Last prompt display showing context used

### Chat Pane

- Text input anchored to the bottom for user messages
- Chronological message display with different styling for user vs. bot
- Support for references display with paper title, author and URL
- Confidence level indicators

### Stats Pane

- Token usage statistics from Gemini response metadata
- Total tokens, prompt tokens, and response tokens
- Current system prompt indicator
- Query type indicator

## Query Types

1. Document queries - search for specific information in papers
2. Aggregation queries - statistics about the collection (paper counts, authors, etc.)
3. Conversational queries - general interaction without document context
4. Composite queries - multiple question types in a single message

## Response Format

- Main answer text
- References to source papers when applicable
- Visual confidence indicator (high/medium/low)
- For composite queries, each sub-query is answered separately