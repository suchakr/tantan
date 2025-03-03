#%%

"""
Document Chat Application with Gemini AI

This program implements a specialized academic document chat system ('Navadhāni') focused on
academic papers from the Indian Journal of History of Science (IJHS). It provides:

1. Document search and retrieval via TF-IDF vectorization and FAISS similarity search
2. Text chunking for effective context management
3. Integration with Google's Gemini API for natural language understanding
4. Structured JSON responses with references and confidence levels
5. Ability to handle both document-specific queries and collection-level statistics
6. A Gradio-based UI for interactive conversations
7. Token usage tracking and conversation history management

The application processes paper content from a TSV file, creates searchable chunks,
and provides context-aware responses using the Gemini model.
"""

#%%
import os
import pandas as pd
import numpy as np
import faiss
import gradio as gr
import google.generativeai as genai
from google.generativeai import GenerationConfig
from dotenv import load_dotenv , find_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import cast, List, Dict, TypedDict
from scipy.sparse import spmatrix
import nltk
from nltk.tokenize import sent_tokenize
import json

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

# logging.basicConfig(level=logging.INFO)
try:
    shell = get_ipython().__class__.__name__
    if shell in ['ZMQInteractiveShell', 'TerminalInteractiveShell']:
        gInteractive = True
    else:
        gInteractive = False
except NameError:
    gInteractive = False

logging.info(f" {gInteractive=}")
#%%
# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

#%%
# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

SYSTEM_PROMPT = """You are Navadhāni, an AI assistant specialized in discussing academic papers from the Indian Journal of History of Science (IJHS).

Your role is to:
1. Provide accurate information based on the academic papers you're given
2. Explain complex concepts in a clear and accessible way
3. Highlight important contributions from Indian scholars in mathematics and astronomy
4. Maintain academic integrity by staying true to the source material
5. Acknowledge when information is not available in the provided papers

Always base your responses on the context provided from the papers. When you refer to any paper, hyperlink the paper to its url.  If asked about topics outside the scope of the given papers, politely explain that you can only discuss content from the IJHS papers in your context."""

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 5,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
    "response_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "references": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "author": {"type": "string"},
                        "url": {"type": "string"}
                    },
                    "required": ["title", "author", "url"]
                }
            },
            "confidence_level": {"type": "string", "enum": ["high", "medium", "low"]}
        },
        "required": ["answer", "references", "confidence_level"]
    }
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
  system_instruction=SYSTEM_PROMPT,
)

# ijhs_astro_math_docs = genai.upload_file(
#     path="ijhs-astro-math-docs.tsv",
#     mime_type="text/plain", 
#     # name="ijhs_astro_math_docs_tsv"
# )
# cache = genai.caching.CachedContent.create(
#   model="gemini-1.5-flash-001",
#   system_instruction=SYSTEM_PROMPT,
#   contents=[ijhs_astro_math_docs],
# )

# genai.caching.CachedContent.list()

# print(f"Model: {cache.model}" , cache.usage_metadata)
# model = genai.GenerativeModel.from_cached_content(
#     cached_content=cache
#     , generation_config= GenerationConfig(**generation_config)
# )

class DocumentChat:
    def __init__(self):
        self.df = None
        self.index = None
        self.vectorizer = None
        self.document_embeddings = None
        self.chunks_df = None  # New DataFrame to store chunks
        self.chunk_size = 500  # Number of characters per chunk
        self.chunk_overlap = 100  # Overlap between chunks
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        self.load_and_process_data()

    def create_chunks(self, text: str) -> List[Dict]:
        if pd.isna(text):
            return []
        
        # Split text into sentences
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Store the current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap
                overlap_point = max(0, len(current_chunk) - 2)  # Keep last 2 sentences for context
                current_chunk = current_chunk[overlap_point:] + [sentence]
                current_length = sum(len(s) for s in current_chunk)
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def load_and_process_data(self):
        try:
            # Load TSV file
            self.df = pd.read_csv('ijhs-astro-math-docs.tsv', sep='\t').drop(columns=[ 'size_in_kb', 'cum_size_in_kb' , 'pdf', 'full_pdf_path', 'pdf_exists', 'has_text'])
            
            # Create chunks for each document
            chunks_data = []
            for idx, row in self.df.iterrows():
                text_chunks = self.create_chunks(row['text'])
                for chunk_idx, chunk in enumerate(text_chunks):
                    chunks_data.append({
                        'paper': row['paper'],
                        'author': row['author'],
                        'url': row['url'],
                        'text': chunk,
                        'original_idx': idx,
                        'chunk_idx': chunk_idx
                    })
            
            self.chunks_df = pd.DataFrame(chunks_data)
            # log df and chunks_df shape
            logging.info(f"Loaded {len(self.df)} documents. Shape: {self.df.shape}. Columns: {self.df.columns}")
            logging.info(f"Created {len(self.chunks_df)} chunks. Shape: {self.chunks_df.shape}. Columns: {self.chunks_df.columns}")
    
            # Initialize and fit TF-IDF vectorizer on chunks
            self.vectorizer = TfidfVectorizer()
            self.document_embeddings = self.vectorizer.fit_transform(self.chunks_df['text'].fillna('')).toarray()
            print(f"Document chunks shape: {self.document_embeddings.shape}")
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.document_embeddings.shape[1])
            self.index.add(self.document_embeddings.astype('float32'))
        except FileNotFoundError:
            print("Warning: ijhs-astro-math-docs.tsv not found. Please ensure the file exists in the current directory.")
            raise

    def search_documents(self, query, k=5):
        # Transform query to vector
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        
        # Search similar chunks
        D, I = self.index.search(query_vector, k)
        
        # Get relevant chunks and deduplicate by original document
        relevant_chunks = [self.chunks_df.iloc[i] for i in I[0]]
        seen_docs = set()
        unique_relevant_chunks = []
        
        for chunk in relevant_chunks:
            doc_id = (chunk['paper'], chunk['original_idx'])
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_relevant_chunks.append(chunk)
        
        return unique_relevant_chunks[:3]  # Limit to top 3 unique documents

    def is_aggregation_query(self, query):
        """Determine if the query is asking for an aggregation/summary of the papers database."""
        query_lower = query.lower()
        
        # Patterns that indicate aggregation queries
        aggregation_patterns = [
            'how many papers', 'how many documents', 'how many articles', 'paper count',
            'list all papers', 'list the papers', 'list papers', 'list all documents',
            'show me all papers', 'show all articles', 'show all documents',
            'what papers do you have', 'what papers do you know', 'what documents are available',
            'summarize the papers', 'summarize the collection', 'summarize the documents',
            'which authors', 'list authors', 'how many authors',
            'most cited', 'most popular', 'most common topics',
            'papers by year', 'papers by author', 'papers by topic',
            'statistics on papers', 'statistics on the collection',
            'what years', 'which years', 'date range', 'year range',
            'overview of papers', 'overview of the collection', 'overview of documents'
        ]
        
        for pattern in aggregation_patterns:
            if pattern in query_lower:
                return True
                
        return False

    def perform_aggregation_naive(self, query):
        """Generate aggregation results based on the query intent using simple pattern matching.
        This is a naive approach and serves as a fallback."""
        query_lower = query.lower()
        
        # Basic collection statistics
        papers_count = len(self.df['paper'].unique()) if self.df is not None else 0
        authors_count = len(self.df['author'].unique()) if self.df is not None else 0
        
        # Initialize the response structure
        result = {
            "answer": "",
            "references": [],
            "confidence_level": "high"  # Aggregations are based on actual data, so confidence is high
        }
        
        # Handle different aggregation types
        if any(p in query_lower for p in ['how many papers', 'paper count', 'document count']):
            result["answer"] = f"I have information about {papers_count} papers from the Indian Journal of History of Science."
            
        elif any(p in query_lower for p in ['list papers', 'list all papers', 'show papers', 'what papers']):
            papers_list = self.df['paper'].unique().tolist() if self.df is not None else []
            papers_text = "\n".join([f"- {paper}" for paper in papers_list])
            result["answer"] = f"Here are the {papers_count} papers in my collection:\n\n{papers_text}"
            
        elif any(p in query_lower for p in ['how many authors', 'author count']):
            result["answer"] = f"There are {authors_count} unique authors in the collection of papers I have access to."
            
        elif any(p in query_lower for p in ['list authors', 'show authors', 'which authors']):
            authors_list = self.df['author'].unique().tolist() if self.df is not None else []
            authors_text = "\n".join([f"- {author}" for author in authors_list])
            result["answer"] = f"Here are the {authors_count} authors in my collection:\n\n{authors_text}"
            
        elif any(p in query_lower for p in ['papers by author', 'which papers by']):
            # Extract author name if it exists in the query
            # This is a simple approach; might need more sophisticated NLP
            author_name = None
            for word in query_lower.split('by '):
                if len(word) > 0 and word not in ['papers', 'author']:
                    author_name = word.strip()
                    break
                    
            if author_name:
                matching_papers = self.df[self.df['author'].str.lower().str.contains(author_name)]['paper'].unique().tolist()
                if matching_papers:
                    papers_text = "\n".join([f"- {paper}" for paper in matching_papers])
                    result["answer"] = f"Here are papers by authors matching '{author_name}':\n\n{papers_text}"
                else:
                    result["answer"] = f"I couldn't find any papers by authors matching '{author_name}' in my collection."
            else:
                # If no author specified, give a breakdown of papers per author
                author_counts = self.df.groupby('author')['paper'].nunique().sort_values(ascending=False)
                author_breakdown = "\n".join([f"- {author}: {count} papers" for author, count in author_counts.items()[:15]])  # Top 15
                result["answer"] = f"Here's a breakdown of papers by author (showing top 15):\n\n{author_breakdown}"
                
        elif 'statistics' in query_lower or 'overview' in query_lower or 'summarize' in query_lower:
            # Provide overall statistics about the collection
            result["answer"] = (
                f"Collection Statistics:\n"
                f"- Total papers: {papers_count}\n"
                f"- Total unique authors: {authors_count}\n"
                f"- Papers with URLs: {self.df['url'].notna().sum()}\n"
                f"- Average text length: {int(self.df['text'].str.len().mean())} characters"
            )
        else:
            # Default aggregation response
            result["answer"] = (
                f"I have a collection of {papers_count} papers from the Indian Journal of History of Science, "
                f"written by {authors_count} different authors. You can ask me to list the papers, "
                f"show papers by a specific author, or get other statistics about the collection."
            )
            
        # Add sample references to guide the user
        if len(self.df) > 0:
            sample_papers = self.df.sample(min(3, len(self.df)))
            for _, paper in sample_papers.iterrows():
                result["references"].append({
                    "title": paper['paper'],
                    "author": paper['author'],
                    "url": paper['url'] if pd.notna(paper['url']) else ""
                })
                
        return result

    def perform_aggregation(self, query):
        """Generate aggregation results by executing dynamic pandas operations based on query intent.
        This approach uses Python data manipulation for more sophisticated analysis."""
        
        if self.df is None or len(self.df) == 0:
            return {
                "answer": "I don't have any document data loaded to analyze.",
                "references": [],
                "confidence_level": "high"
            }
            
        # Initialize result structure
        result = {
            "answer": "",
            "references": [],
            "confidence_level": "high"
        }
        
        query_lower = query.lower()
        
        try:
            # Basic counts and lists
            if any(p in query_lower for p in ['how many papers', 'paper count', 'document count']):
                paper_count = self.df['paper'].nunique()
                result["answer"] = f"I have information about {paper_count} papers from the Indian Journal of History of Science."
                
            elif any(p in query_lower for p in ['list papers', 'list all papers', 'show papers', 'what papers']):
                papers = self.df['paper'].unique().tolist()
                papers_text = "\n".join([f"- {paper}" for paper in papers])
                result["answer"] = f"Here are the {len(papers)} papers in my collection:\n\n{papers_text}"
                
            elif any(p in query_lower for p in ['how many authors', 'author count']):
                author_count = self.df['author'].nunique()
                result["answer"] = f"There are {author_count} unique authors in the collection of papers I have access to."
                
            elif any(p in query_lower for p in ['list authors', 'show authors', 'which authors']):
                authors = self.df['author'].unique().tolist()
                authors_text = "\n".join([f"- {author}" for author in authors])
                result["answer"] = f"Here are the {len(authors)} authors in my collection:\n\n{authors_text}"
                
            # Papers by specific author
            elif any(p in query_lower for p in ['papers by author', 'which papers by']):
                # Extract potential author name from query
                author_name = None
                for pattern in ['by author', 'by', 'author']:
                    if pattern in query_lower:
                        parts = query_lower.split(pattern)
                        if len(parts) > 1 and parts[1].strip():
                            author_name = parts[1].strip()
                            break
                
                if author_name:
                    # Use case-insensitive search
                    matching_papers = self.df[self.df['author'].str.lower().str.contains(author_name)]
                    
                    if not matching_papers.empty:
                        papers_by_author = matching_papers.groupby('author')['paper'].unique()
                        
                        answer_parts = [f"Papers by authors matching '{author_name}':"]
                        for author, papers in papers_by_author.items():
                            answer_parts.append(f"\n{author}:")
                            for paper in papers:
                                answer_parts.append(f"- {paper}")
                        
                        result["answer"] = "\n".join(answer_parts)
                        
                        # Add matching papers as references
                        for _, row in matching_papers.drop_duplicates(subset=['paper']).iterrows():
                            result["references"].append({
                                "title": row['paper'],
                                "author": row['author'],
                                "url": row['url'] if pd.notna(row['url']) else ""
                            })
                    else:
                        result["answer"] = f"I couldn't find any papers by authors matching '{author_name}' in my collection."
                else:
                    # No specific author mentioned, show distribution
                    author_paper_counts = self.df.groupby('author')['paper'].nunique().sort_values(ascending=False)
                    top_authors = author_paper_counts.head(15)
                    
                    answer_parts = ["Here's a breakdown of papers by author (showing top 15):"]
                    for author, count in top_authors.items():
                        answer_parts.append(f"- {author}: {count} papers")
                    
                    result["answer"] = "\n".join(answer_parts)
                    
            # Overall collection statistics
            elif 'statistics' in query_lower or 'overview' in query_lower or 'summarize' in query_lower:
                # Calculate various statistics
                text_lengths = self.df['text'].str.len()
                url_count = self.df['url'].notna().sum()
                
                # Create detailed statistics
                stats = {
                    "Total papers": self.df['paper'].nunique(),
                    "Unique authors": self.df['author'].nunique(),
                    "Papers with URLs": url_count,
                    "Average text length": int(text_lengths.mean()),
                    "Longest paper": int(text_lengths.max()),
                    "Shortest paper": int(text_lengths.min()),
                    "Authors with most papers": self.df.groupby('author')['paper'].nunique().nlargest(3).to_dict()
                }
                
                answer_parts = ["Collection Statistics:"]
                for key, value in stats.items():
                    if isinstance(value, dict):
                        answer_parts.append(f"- {key}:")
                        for subkey, subvalue in value.items():
                            answer_parts.append(f"  - {subkey}: {subvalue} papers")
                    else:
                        if "length" in key.lower():
                            answer_parts.append(f"- {key}: {value} characters")
                        else:
                            answer_parts.append(f"- {key}: {value}")
                
                result["answer"] = "\n".join(answer_parts)
                
            else:
                # Fall back to the naive implementation for unhandled query types
                return self.perform_aggregation_naive(query)
                
            # Add sample references if we don't have specific ones yet
            if not result["references"] and len(self.df) > 0:
                sample_papers = self.df.sample(min(3, len(self.df)))
                for _, paper in sample_papers.iterrows():
                    result["references"].append({
                        "title": paper['paper'],
                        "author": paper['author'],
                        "url": paper['url'] if pd.notna(paper['url']) else ""
                    })
                    
            return result
            
        except Exception as e:
            logging.warning(f"Error in advanced aggregation: {str(e)}. Falling back to naive implementation.")
            return self.perform_aggregation_naive(query)

    def needs_document_context(self, query):
        """Determine if a query needs document context or can be answered directly."""
        # First check if this is an aggregation query
        if self.is_aggregation_query(query):
            return False
            
        # Simple keywords that might indicate we need papers context
        document_keywords = [
            'paper', 'research', 'study', 'article', 'publication', 'journal', 
            'publish', 'author', 'work', 'contribution', 'discover', 'theory',
            'mathematician', 'astronomy', 'history', 'science', 'indian', 'ijhs',
            'who', 'what', 'when', 'where', 'how', 'why', 'did', 'explain'
        ]
        
        # Direct questions that might not need context
        direct_patterns = [
            'who are you', 'your name', 'what can you do', 'hello', 'hi',
            'help me', 'how do you work', 'what is your purpose',
            'thanks', 'thank you'
        ]
        
        # Check for direct patterns first
        query_lower = query.lower()
        for pattern in direct_patterns:
            if pattern in query_lower:
                return False
                
        # Check for document keywords
        for keyword in document_keywords:
            if keyword.lower() in query_lower:
                return True
                
        # If query is very short, it might be conversational
        if len(query.split()) < 4:
            return False
            
        # Default to using context for ambiguous queries
        return True

    def generate_response(self, query):
        try:
            # Check if this is an aggregation query
            if self.is_aggregation_query(query):
                # Handle aggregation directly without LLM
                return self.perform_aggregation(query), None
            
            # For non-aggregation queries, continue with the existing flow
            elif self.needs_document_context(query):
                # Get relevant documents
                relevant_chunks = self.search_documents(query)
                
                # Prepare context with chunked content
                context = "\n".join([
                    f"Title: {chunk['paper']}\nAuthor: {chunk['author']}\nURL: {chunk['url']}\nRelevant Extract: {chunk['text']}"
                    for chunk in relevant_chunks
                ])
        
                # Generate response using Gemini with system prompt and document context
                query_obj = {
                    "query": query,
                    "require_context": True,
                    "context": {
                        "documents": [
                            {
                                "title": chunk['paper'],
                                "author": chunk['author'],
                                "url": chunk['url'],
                                "extract": chunk['text']
                            }
                            for chunk in relevant_chunks
                        ]
                    }
                }
                
                prompt = f"""Based on the following relevant extracts from IJHS papers, answer this query: {query}

Context from relevant papers:
{context}

Please provide a clear and concise response based on the information from these papers.
Your response must conform to the JSON schema provided in the system instructions."""
            else:
                # Generate response without document context for general/conversational queries
                query_obj = {
                    "query": query,
                    "require_context": False
                }
                
                prompt = f"""Please respond to this query without needing additional context: {query}
Your response must conform to the JSON schema provided in the system instructions."""
            
            response = model.generate_content(prompt)
            
            # Parse the JSON response
            try:
                if hasattr(response, 'parts') and len(response.parts) > 0 and hasattr(response.parts[0], 'text'):
                    response_json = json.loads(response.parts[0].text)
                else:
                    response_json = json.loads(response.text)
                
                # Return structured data and usage metadata
                return response_json, getattr(response, 'usage_metadata', None)
            except json.JSONDecodeError:
                # Fallback for non-JSON responses
                logging.warning("Received non-JSON response from LLM")
                fallback_response = {
                    "answer": response.text,
                    "references": [],
                    "confidence_level": "low"
                }
                return fallback_response, getattr(response, 'usage_metadata', None)
        
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            error_response = {
                "answer": f"Error: {str(e)}",
                "references": [],
                "confidence_level": "low"
            }
            return error_response, None
        
#%%
# Run this block to test the DocumentChat class and its methods
def test_DocumentChat():
    chat = DocumentChat()
    query = "What are the contributions of Indian mathematicians to the field of astronomy?"
    # print(chat.search_documents(query))
    response, usage_metadata = chat.generate_response(query)
    print(response)
    print(usage_metadata)

def test_aggregation_queries():
    chat = DocumentChat()
    queries = [
        "How many papers do you have?",
        "List all papers in your collection",
        "How many authors are there?",
        "List authors in your collection",
        "What papers do you have by author Aryabhata?",
        "Summarize the collection",
        "Show me statistics on the papers",
        "What years are covered in the collection?",
        "Overview of the documents"
    ]
    
    for query in queries:
        print(f"Query: {query}")
        response = chat.perform_aggregation(query)
        print(response)
        print() # Add a newline

if gInteractive:
    test_DocumentChat()
    test_aggregation_queries()



#%%
class NavdhaniUI:
    def __init__(self):
        self.chat = DocumentChat()
        self.history = []  # Maintain conversation history here
        self.token_stats = {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0}
        self.last_prompt = ""
        self.context_used = False  # Track whether context was used for last response

    def update_stats(self, usage_metadata):
        if usage_metadata:
            try:
                prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
                total_tokens = getattr(usage_metadata, 'total_token_count', 0)

                self.token_stats["prompt_tokens"] += prompt_tokens
                self.token_stats["response_tokens"] += completion_tokens
                self.token_stats["total_tokens"] += total_tokens

            except Exception as e:
                print(f"Error updating stats: {str(e)}")

        context_status = "With document context" if self.context_used else "Without document context"
        
        return f"""Token Usage Statistics:
Total Tokens: {self.token_stats['total_tokens']}
Prompt Tokens: {self.token_stats['prompt_tokens']}
Response Tokens: {self.token_stats['response_tokens']}
Last Query: {context_status}"""

    def respond(self, message, chat_history):
        # Check if document context is needed
        self.context_used = self.chat.needs_document_context(message)
        
        # Generate response and update internal history
        response_data, usage_metadata = self.chat.generate_response(message)
        stats = self.update_stats(usage_metadata)
        
        # Format the response for display
        formatted_response = response_data["answer"]
        
        # Add references if they exist
        if response_data.get("references") and len(response_data["references"]) > 0:
            formatted_response += "\n\nReferences:"
            for ref in response_data["references"]:
                formatted_response += f"\n- {ref['title']} by {ref['author']} - {ref['url']}"
        
        # Add confidence level
        if response_data.get("confidence_level"):
            formatted_response += f"\n\nConfidence: {response_data['confidence_level']}"
        
        # Update self.history
        self.history.append((message, formatted_response))
        
        # Update last prompt with the full context if used
        if self.context_used:
            relevant_docs = self.chat.search_documents(message)
            context = "\n".join([
                f"Title: {doc['paper']}\nAuthor: {doc['author']}\nContent: {doc['text'][:500]}..."
                for doc in relevant_docs
            ])
            self.last_prompt = f"""Based on the following academic papers from IJHS, answer this query: {message}

Context from relevant papers:
{context}

Please provide a clear and concise response based on the information from these papers in JSON format."""
        else:
            self.last_prompt = f"""Please respond to this query without needing additional context: {message} in JSON format."""
        
        return self.history, stats, self.last_prompt

    def reset_conversation(self):
        # Reset conversation state
        self.history = []
        self.token_stats = {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0}
        self.last_prompt = ""
        self.context_used = False
        # Return cleared values to update the UI
        cleared_stats = "Token Usage Statistics:\nTotal Tokens: 0\nPrompt Tokens: 0\nResponse Tokens: 0"
        return self.history, cleared_stats, "No prompt sent yet"

    def launch(self):
        with gr.Blocks() as interface:
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=600)
                    msg = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message here...",
                    )
                    # Add a reset button for starting a new conversation
                    reset_btn = gr.Button("New Conversation")
                    
                with gr.Column(scale=1):
                    stats_display = gr.Textbox(
                        label="Statistics",
                        interactive=False,
                        value="Token Usage Statistics:\nTotal Tokens: 0\nPrompt Tokens: 0\nResponse Tokens: 0"
                    )
                    prompt_display = gr.Textbox(
                        label="Last Prompt",
                        interactive=False,
                        value="No prompt sent yet",
                        lines=10
                    )

            # When a message is submitted, call respond
            msg.submit(
                self.respond,
                [msg, chatbot],
                [chatbot, stats_display, prompt_display]
            )
            # Bind the reset button to reset_conversation. It updates the chatbot, stats, and prompt.
            reset_btn.click(
                self.reset_conversation,
                None,
                [chatbot, stats_display, prompt_display]
            )

        interface.launch()

if __name__ == "__main__":
    ui = NavdhaniUI()
    ui.launch()

# %%
# # Define response schema for structured output
# class PaperReference(TypedDict):
#     title: str
#     author: str
#     url: str

# class StructuredResponse(TypedDict):
#     answer: str
#     references: List[PaperReference]
#     confidence_level: str  # "high", "medium", "low"
