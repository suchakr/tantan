#%%

"""
Document Chat Application with Gemini AI ('Navadhāni')

A specialized academic document chat system focused on papers from the Indian Journal of History of Science (IJHS).
Features:
1. Document search and retrieval via TF-IDF vectorization and FAISS similarity search
2. Text chunking for effective context management
3. Integration with Google's Gemini API for natural language understanding
4. Structured JSON responses with references and confidence levels
5. Ability to handle both document-specific queries and collection-level statistics
6. Gradio-based UI for interactive conversations
7. Token usage tracking and conversation history management
8. Composite message handling with prompt splitting
"""

import os
import json
import time
import logging
from typing import List, Dict, TypedDict, Optional, Tuple, Any, cast, Union
from dataclasses import dataclass

# Data processing imports
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
import numpy.typing as npt
from scipy.sparse import spmatrix, csr_matrix
import scipy.sparse as sp

# UI and API imports
import gradio as gr
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv, find_dotenv

# Import system prompts
from system_prompts import get_prompt

# Import LLMAggregationHandler for advanced aggregation capabilities
from llm_aggregation_handler import LLMAggregationHandler

# Import prompt splitter for composite message handling
from prompt_splitter import HybridPromptSplitter, PromptSplitter, HeuristicPromptSplitter, QueryType

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


# Load environment variables and configure API
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

# Gemini model configuration
generation_config = GenerationConfig(
    temperature=1.0,
    top_p=0.95,
    top_k=5,
    max_output_tokens=8192,
)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document with metadata"""
    paper: str
    author: str
    url: str
    text: str
    original_idx: int
    chunk_idx: int

class DocumentChat:
    """Handles document processing, searching, and response generation"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, prompt_name: str = "" ,
                 tsv_file: str='ijhs-astro-math-docs.tsv'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.df: Optional[pd.DataFrame] = None
        self.chunks_df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.document_embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatL2] = None
        self.query_type: str = "document"
        
        # Get the system prompt based on the provided name (or default if None)
        system_prompt = get_prompt(prompt_name)
        
        # Initialize Gemini model with the selected system prompt
        genai.configure(api_key=GOOGLE_API_KEY)  # type: ignore
        self.model = genai.GenerativeModel(      # type: ignore
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
            system_instruction=system_prompt,
        )
        
        # Initialize NLTK
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        self.tsv_file = tsv_file
        self.agg_handler = LLMAggregationHandler(dataset_path=self.tsv_file)
        self.load_and_process_data()

    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks for better context management"""
        if pd.isna(text):
            return []
        
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
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                overlap_point = max(0, len(current_chunk) - 2)
                current_chunk = current_chunk[overlap_point:] + [sentence]
                current_length = sum(len(s) for s in current_chunk)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def load_and_process_data(self):
        """Load and process the document data, creating searchable chunks"""
        try:
            # Load the TSV file and drop unnecessary columns
            self.df = pd.read_csv(self.tsv_file, sep='\t').drop(
                columns=['size_in_kb', 'cum_size_in_kb', 'pdf', 'full_pdf_path', 'pdf_exists', 'has_text']
            )
            # print(self.df.shape)
            self.df = self.df.dropna(subset=['text'])
            # print(self.df.shape)
            # self.df['text'] = self.df['text'].str.replace('\n', ' ')
            
            # Create chunks for each document
            chunks_data = []
            for idx, row in self.df.iterrows():
                # print(row)
                text_chunks = self.create_chunks(
                    row['author'] 
                    + " " + row['paper'] 
                    + " " + row['text']
                    )
                for chunk_idx, chunk in enumerate(text_chunks):
                    chunks_data.append(DocumentChunk(
                        paper=row['paper'],
                        author=row['author'],
                        url=row['url'],
                        text=chunk,
                        original_idx=int(cast(int, idx)),
                        chunk_idx=chunk_idx
                    ).__dict__)
            
            self.chunks_df = pd.DataFrame(chunks_data)
            logging.info(f"Loaded {len(self.df)} documents, created {len(self.chunks_df)} chunks")
            
            # Initialize search components with proper typing
            self.vectorizer = TfidfVectorizer()
            sparse_matrix = self.vectorizer.fit_transform(self.chunks_df['text'].fillna(''))
            self.document_embeddings = sparse_matrix.toarray().astype(np.float32)
            
            dim = self.document_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            if self.document_embeddings is not None:
                self.index.add(self.document_embeddings)
            logging.info("Search components initialized successfully")
            
        except FileNotFoundError:
            logging.error(f"{tsv_file} not found")
            raise

    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant document chunks based on query similarity"""
        if not isinstance(self.vectorizer, TfidfVectorizer) or self.index is None or self.chunks_df is None:
            raise ValueError("Search components not initialized")
            
        query_matrix: csr_matrix = self.vectorizer.transform([query])  # type: ignore
        query_vector = query_matrix.toarray().astype(np.float32)
        
        # Use standard FAISS search API with proper return values
        distances, indices = self.index.search(query_vector, k)
        
        # Get relevant chunks and deduplicate by original document
        relevant_chunks = [self.chunks_df.iloc[i].to_dict() for i in indices[0]]
        seen_docs = set()
        unique_relevant_chunks = []
        
        for chunk in relevant_chunks:
            doc_id = (chunk['paper'], chunk['original_idx'])
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                unique_relevant_chunks.append(chunk)
        
        return unique_relevant_chunks[:3]

    def is_aggregation_query(self, query: str) -> bool:
        """Determine if the query is asking for collection-level statistics"""
        query_lower = query.lower()
        
        aggregation_patterns = [
            'how many papers', 'how many documents', 'paper count',
            'list all papers', 'list papers', 'show papers',
            'what papers do you have', 'what documents are available',
            'summarize the papers', 'summarize the collection',
            'which authors', 'list authors', 'how many authors',
            'most cited', 'most popular', 'most common topics',
            'papers by year', 'papers by author', 'papers by topic',
            'statistics', 'overview', 'what years', 'date range'
        ]
        
        return any(pattern in query_lower for pattern in aggregation_patterns)
    
    def is_conversational_query(self, query: str) -> bool:
        """Determine if the query is conversational in nature"""
        conversational_patterns = [
            'who are you', 'your name', 'what can you do', 'hello', 'hi',
            'help me', 'how do you work', 'what is your purpose',
            'thanks', 'thank you'
        ]

        return any(pattern in query.lower() for pattern in conversational_patterns
                  ) and not self.is_aggregation_query(query)

    def needs_document_context(self, query: str) -> bool:
        """Determine if a query requires document context for answering"""

        # Check for conversational queries
        if self.is_conversational_query(query):
            self.query_type = "conversational"
            return False

        # Check for aggregation queries
        # These queries often require collection-level statistics
        if self.is_aggregation_query(query):
            self.query_type = "aggregation"
            return False
    
        self.query_type = "document"
        return True

    def perform_aggregation(self, query: str) -> Dict:
        """Generate collection-level statistics and aggregations"""
        if self.df is None or self.df.empty:
            return {
                "answer": "No document data available for analysis.",
                "references": [],
                "confidence_level": "high"
            }
        
        query_lower = query.lower()
        result = {
            "answer": "",
            "references": [],
            "confidence_level": "high"
        }
        
        try:
            # Basic collection statistics
            if 'statistics' in query_lower or 'overview' in query_lower :
                stats = {
                    "Total papers": self.df['paper'].nunique(),
                    "Unique authors": self.df['author'].nunique(),
                    "Papers with URLs": self.df['url'].notna().sum(),
                    "Average text length": int(self.df['text'].str.len().mean()),
                    "Most prolific authors": self.df.groupby('author')['paper'].nunique().nlargest(3).to_dict()
                }
                
                result["answer"] = "Collection Statistics:\n" + "\n".join(
                    f"- {key}: {value}" if not isinstance(value, dict) else
                    f"- {key}:\n" + "\n".join(f"  - {k}: {v} papers" for k, v in value.items())
                    for key, value in stats.items()
                )
            
            # Author-specific queries
            elif any(x in query_lower for x in ['author', 'who wrote']):
                author_query = next((
                    part.strip()
                    for pattern in ['by author', 'by', 'author']
                    for part in query_lower.split(pattern)[1:]
                    if part.strip()
                ), None)
                
                if author_query:
                    matches = self.df[self.df['author'].str.lower().str.contains(author_query)]
                    if not matches.empty:
                        papers_by_author = matches.groupby('author')['paper'].unique()
                        result["answer"] = "\n\n".join(
                            f"{author}:\n" + "\n".join(f"- {paper}" for paper in papers)
                            for author, papers in papers_by_author.items()
                        )
                    else:
                        result["answer"] = f"No papers found by authors matching '{author_query}'"
                else:
                    top_authors = self.df.groupby('author')['paper'].nunique().sort_values(ascending=False)
                    result["answer"] = "Top authors by number of papers:\n" + "\n".join(
                        f"- {author}: {count} papers"
                        for author, count in top_authors.head(10).items()
                    )
            
            # Paper listing and counting
            elif any(x in query_lower for x in ['list papers', 'list all papers', 'show papers', 'list paper titles', 'list all paper titles', 'how many papers']):
                paper_count = self.df['paper'].nunique()
                
                # Specific handling for listing papers
                if 'list' in query_lower or 'show' in query_lower:
                    papers_list = self.df[['paper', 'author', 'url']].drop_duplicates('paper').sort_values('paper')
                    
                    # Format the list of papers with their authors and URLs if available
                    papers_formatted = []
                    for _, row in papers_list.iterrows():
                        paper_str = f"- {row['paper']} by {row['author']}"
                        if pd.notna(row['url']):
                            paper_str += f" - {row['url']}"
                        papers_formatted.append(paper_str)
                    
                    result["answer"] = f"Here are all {paper_count} papers in the collection:\n\n" + "\n".join(papers_formatted)
                    
                    # Add all papers as references
                    result["references"] = [
                        {
                            "title": row['paper'],
                            "author": row['author'],
                            "url": row['url'] if pd.notna(row['url']) else ""
                        }
                        for _, row in papers_list.iterrows()
                    ]
                else:
                    result["answer"] = f"The collection contains {paper_count} papers."
                    
                    # Add sample papers as references if we're not listing all of them
                    sample_papers = self.df.drop_duplicates('paper').sample(min(3, paper_count))
                    result["references"] = [
                        {
                            "title": row['paper'],
                            "author": row['author'],
                            "url": row['url'] if pd.notna(row['url']) else ""
                        }
                        for _, row in sample_papers.iterrows()
                    ]
            
            else:
                return self.perform_aggregation_naive(query)
            
            # If we didn't add references earlier and we're not listing all papers,
            # add sample references
            if not result["references"] and not self.df.empty:
                sample_papers = self.df.drop_duplicates('paper').sample(min(3, len(self.df)))
                result["references"] = [
                    {
                        "title": row['paper'],
                        "author": row['author'],
                        "url": row['url'] if pd.notna(row['url']) else ""
                    }
                    for _, row in sample_papers.iterrows()
                ]
            
            return result
            
        except Exception as e:
            logging.warning(f"Error in aggregation: {str(e)}. Falling back to naive implementation.")
            return self.perform_aggregation_naive(query)

    def perform_aggregation_naive(self, query: str) -> Dict:
        """Fallback method for aggregation queries not handled by specific rules"""
        # This method isn't defined in the original code, so adding a simple implementation
        try:
            if self.df is None or self.df.empty:
                return {
                    "answer": "No document data available for analysis.",
                    "references": [],
                    "confidence_level": "high"
                }
                
            # Generate a simple collection overview
            paper_count = self.df['paper'].nunique()
            author_count = self.df['author'].nunique()
            
            answer = f"The collection contains {paper_count} papers by {author_count} authors. "
            answer += "You can ask for specific statistics, paper listings, or author information."
            
            # Add sample references
            sample_papers = self.df.drop_duplicates('paper').sample(min(3, len(self.df)))
            references = [
                {
                    "title": row['paper'],
                    "author": row['author'],
                    "url": row['url'] if pd.notna(row['url']) else ""
                }
                for _, row in sample_papers.iterrows()
            ]
            
            return {
                "answer": answer,
                "references": references,
                "confidence_level": "medium"
            }
            
        except Exception as e:
            logging.error(f"Error in naive aggregation: {str(e)}")
            return {
                "answer": f"Unable to process aggregation query: {str(e)}",
                "references": [],
                "confidence_level": "low"
            }

    def generate_response(self, query: str) -> Tuple[Dict, Any]:
        """Generate a response to a user query using appropriate context and model"""
        try:
            if self.is_aggregation_query(query):
                # Use LLMAggregationHandler for advanced aggregation processing
                try:
                    response = self.agg_handler.process_query(query)
                    usage_metadata = response.get('usage_metadata')
                    
                    # # Remove generated_code and usage_metadata from the response
                    # if 'generated_code' in response:
                    #     del response['generated_code']
                    # if 'usage_metadata' in response:
                    #     del response['usage_metadata']
                        
                    return response, usage_metadata
                except Exception as e:
                    logging.warning(f"Error with LLM aggregation handler: {str(e)}. Falling back to basic aggregation.")
                    # Fall back to the original method
                    return self.perform_aggregation(query), None
            
            if self.needs_document_context(query):
                relevant_chunks = self.search_documents(query)
                context = "\n".join([
                    f"Title: {chunk['paper']}\nAuthor: {chunk['author']}\n"
                    f"URL: {chunk['url']}\nRelevant Extract: {chunk['text']}"
                    for chunk in relevant_chunks
                ])
                
                prompt = f"""Based on these IJHS papers, answer this query: {query}

Context from relevant papers:
{context}

Please provide a clear and concise response based on these papers."""
            else:
                prompt = f"Please respond to this query: {query}"
            
            response = self.model.generate_content(prompt)
            
            try:
                response_json = (
                    json.loads(response.parts[0].text)
                    if hasattr(response, 'parts') and response.parts
                    else json.loads(response.text)
                )
                return response_json, getattr(response, 'usage_metadata', None)
            except json.JSONDecodeError:
                return {
                    "answer": response.text,
                    "references": [],
                    "confidence_level": "low"
                }, getattr(response, 'usage_metadata', None)
                
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return {
                "answer": f"Error: {str(e)}",
                "references": [],
                "confidence_level": "low"
            }, None
        

#%%
class NavdhaniUI:
    """Handles the Gradio-based user interface and conversation management"""
    
    def __init__(self, prompt_name=""):
        try:
            self.chat = DocumentChat(prompt_name=prompt_name)
        except Exception as e:
            logging.error(f"Failed to initialize DocumentChat: {e}")
            raise
            
        self.history: List[Tuple[str, str]] = []
        self.token_stats: Dict[str, int] = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "response_tokens": 0
        }
        self.last_prompt: str = ""
        self.context_used: bool = False
        self.query_type: str = "document"
        self.prompt_name = prompt_name or "default"
        
        # Initialize the prompt splitter for handling composite messages
        # Use HeuristicPromptSplitter by default (no API key needed)
        self.prompt_splitter = HeuristicPromptSplitter()
        # Try to initialize a HybridPromptSplitter if an API key is available
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                self.prompt_splitter = HybridPromptSplitter(api_key)
                logging.info("Using HybridPromptSplitter for composite message handling")
            else:
                logging.info("Using HeuristicPromptSplitter for composite message handling (no API key)")
        except Exception as e:
            logging.warning(f"Failed to initialize HybridPromptSplitter: {e}. Using HeuristicPromptSplitter instead.")
            
    def update_stats(self, usage_metadata: Optional[Any]) -> str:
        """Update token usage statistics and return formatted stats string"""
        try:
            if usage_metadata:
                self.token_stats["prompt_tokens"] += getattr(usage_metadata, 'prompt_token_count', 0)
                self.token_stats["response_tokens"] += getattr(usage_metadata, 'candidates_token_count', 0)
                self.token_stats["total_tokens"] = (
                    self.token_stats["prompt_tokens"] + 
                    self.token_stats["response_tokens"]
                )
        except Exception as e:
            logging.warning(f"Error updating token stats: {str(e)}")
        # context_status = "With document context" if self.context_used else "Without document context"
        return f"""Token Usage Statistics:
Total Tokens: {self.token_stats['total_tokens']}
Prompt Tokens: {self.token_stats['prompt_tokens']}
Response Tokens: {self.token_stats['response_tokens']}
System Prompt: {self.prompt_name}
Query Type: {self.query_type}"""

    def respond(
        self, 
        message: str, 
        chat_history: List[Tuple[str, str]]
    ) -> Tuple[List[Tuple[str, str]], str, str]:
        """Generate a response to user input and update conversation state"""
        try:
            if not message.strip():
                return self.history, self.update_stats(None), "Empty query"

            # Use prompt splitter to handle composite messages
            grouped_queries = self.prompt_splitter.get_grouped_queries(message)
            
            # Log the split queries for debugging
            for query_type, queries in grouped_queries.items():
                if queries:
                    logging.info(f"Found {len(queries)} queries of type {query_type.name}")
                    
            # Initialize response components
            final_responses = []
            combined_references = []
            all_usage_metadata = []
            confidence_levels = []
            contexts_used = []
            generated_code = None
            
            # Process each query group in a logical order (conversational -> format -> aggregation -> document)
            for query_type in [QueryType.CONVERSATIONAL, QueryType.FORMAT, QueryType.AGGREGATION, QueryType.DOCUMENT, QueryType.UNKNOWN]:
                queries = grouped_queries.get(query_type, [])
                
                if not queries:
                    continue
                
                for query in queries:
                    # Determine context needs based on query type
                    self.context_used = self.chat.needs_document_context(query)
                    self.query_type = self.chat.query_type
                    contexts_used.append(self.context_used)
                    
                    # Generate response for this specific query
                    response_data, usage_metadata = self.chat.generate_response(query)
                    
                    # Collect response components
                    if response_data.get("answer"):
                        final_responses.append(response_data["answer"])
                    
                    if response_data.get("references"):
                        combined_references.extend(response_data["references"])
                    
                    if response_data.get("confidence_level"):
                        confidence_levels.append(response_data["confidence_level"])
                        
                    if usage_metadata:
                        all_usage_metadata.append(usage_metadata)
                        
                    if response_data.get("generated_code") and not generated_code:
                        generated_code = response_data.get("generated_code")
            
            # Format the overall response
            final_response = "\n\n".join(final_responses)
            
            # Add references if any context was used
            if any(contexts_used) and combined_references:
                # Deduplicate references by title
                seen_titles = set()
                unique_references = []
                for ref in combined_references:
                    title = ref.get("title", "")
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        unique_references.append(ref)
                
                # Add references section
                if unique_references:
                    final_response += "\n\nReferences:"
                    for ref in unique_references:
                        if all(ref.get(k) for k in ["title", "author", "url"]):
                            final_response += f"\n- {ref['title']} by {ref['author']} - {ref['url']}"
            
            # Add overall confidence level
            if confidence_levels:
                # Use the lowest confidence level
                confidence_map = {"high": 3, "medium": 2, "low": 1}
                confidence_values = [confidence_map.get(level, 0) for level in confidence_levels]
                overall_confidence = "low"
                if confidence_values:
                    min_confidence = min(confidence_values)
                    if min_confidence == 3:
                        overall_confidence = "high"
                    elif min_confidence == 2:
                        overall_confidence = "medium"
                
                final_response += f"\n\nConfidence: {overall_confidence}"
            
            # Update conversation history
            self.history.append((message, final_response))
            
            # Update stats for all usage metadata
            stats_display = self.update_stats_from_multiple(all_usage_metadata)
            
            # Format the prompt display
            prompt_display = self.format_prompt_display(message, grouped_queries)
            
            return self.history, stats_display, prompt_display
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logging.error(error_msg)
            self.history.append((message, error_msg))
            return self.history, self.update_stats(None), "Error occurred during processing"
    
    def update_stats_from_multiple(self, usage_metadata_list: List[Any]) -> str:
        """Update token usage statistics from multiple metadata objects"""
        try:
            for metadata in usage_metadata_list:
                if metadata:
                    self.token_stats["prompt_tokens"] += getattr(metadata, 'prompt_token_count', 0)
                    self.token_stats["response_tokens"] += getattr(metadata, 'candidates_token_count', 0)
                    
            self.token_stats["total_tokens"] = (
                self.token_stats["prompt_tokens"] + 
                self.token_stats["response_tokens"]
            )
            
        except Exception as e:
            logging.warning(f"Error updating token stats: {str(e)}")
            
        return f"""Token Usage Statistics:
Total Tokens: {self.token_stats['total_tokens']}
Prompt Tokens: {self.token_stats['prompt_tokens']}
Response Tokens: {self.token_stats['response_tokens']}
System Prompt: {self.prompt_name}
Query Type: {"composite" if len(usage_metadata_list) > 1 else self.query_type}"""

    def format_prompt_display(self, message: str, grouped_queries: Dict[QueryType, List[str]]) -> str:
        """Format the prompt display with information about split queries"""
        try:
            # Count the total number of queries
            total_queries = sum(len(queries) for queries in grouped_queries.values())
            
            if total_queries <= 1:
                # Simple case: just one query
                try:
                    if self.context_used:
                        relevant_docs = self.chat.search_documents(message)
                        context = "\n\n=============\n".join(
                            f"Title: {doc['paper']}\nAuthor: {doc['author']}\n"
                            f"Content: {doc['text'][:200]}..."
                            for doc in relevant_docs
                        )
                        return f"""Query with context: {message}\n\nRelevant papers:\n{context}"""
                    else:
                        return f"Direct query: {message}"
                except Exception as e:
                    logging.warning(f"Error formatting simple prompt display: {e}")
                    return f"Query: {message}"
            else:
                # Complex case: multiple queries
                display = f"Composite query detected: {message}\n\nSplit into {total_queries} sub-queries:\n"
                
                for query_type, queries in grouped_queries.items():
                    if not queries:
                        continue
                        
                    display += f"\n{query_type.name} queries ({len(queries)}):\n"
                    for i, query in enumerate(queries):
                        display += f"{i+1}. {query}\n"
                
                return display
                
        except Exception as e:
            logging.warning(f"Error formatting prompt display: {e}")
            return f"Query: {message}"
            
    def reset_conversation(self) -> Tuple[List[Tuple[str, str]], str, str]:
        """Reset the conversation state and statistics"""
        self.history = []
        self.token_stats = {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0}
        self.last_prompt = ""
        self.context_used = False
        return [], f"Token Usage Statistics:\nTotal Tokens: 0\nPrompt Tokens: 0\nResponse Tokens: 0\nSystem Prompt: {self.prompt_name}", "No prompt sent yet"
        
    def change_prompt(self, prompt_name: str) -> Tuple[List[Tuple[str, str]], str, str]:
        """Change the system prompt and reset the conversation"""
        try:
            self.chat = DocumentChat(prompt_name=prompt_name)
            self.prompt_name = prompt_name
            return self.reset_conversation()
        except Exception as e:
            logging.error(f"Failed to change system prompt: {e}")
            error_msg = f"Failed to change system prompt: {str(e)}"
            self.history.append(("System", error_msg))
            return self.history, self.update_stats(None), "Error occurred while changing prompt"

    def launch(self, share: bool = False):
        """Launch the Gradio interface with proper error handling"""
        try:
            with gr.Blocks() as interface:
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            height=600,
                            show_label=True,
                            label="Navadhāni: Indian History of Science Chatbot",
                        )
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Type your message here...",
                        )
                        with gr.Row():
                            reset_btn = gr.Button("New Conversation")
                            
                            # Add dropdown for system prompt selection
                            prompt_selector = gr.Dropdown(
                                choices=["default", "conversational", "expert"],
                                value="default",
                                label="System Prompt",
                            )
                        
                    with gr.Column(scale=1):
                        stats_display = gr.Textbox(
                            label="Statistics",
                            interactive=False,
                            value=f"Token Usage Statistics:\nTotal Tokens: 0\nPrompt Tokens: 0\nResponse Tokens: 0\nSystem Prompt: {self.prompt_name}"
                        )
                        prompt_display = gr.Textbox(
                            label="Last Prompt",
                            interactive=False,
                            value="No prompt sent yet",
                            lines=10
                        )

                msg.submit(
                    self.respond,
                    [msg, chatbot],
                    [chatbot, stats_display, prompt_display]
                )
                reset_btn.click(
                    self.reset_conversation,
                    None,
                    [chatbot, stats_display, prompt_display]
                )
                prompt_selector.change(
                    self.change_prompt,
                    [prompt_selector],
                    [chatbot, stats_display, prompt_display]
                )

            interface.launch(share=share)
            
        except Exception as e:
            logging.error(f"Failed to launch UI: {e}")
            raise

def run_tests():
    """Run test cases for the DocumentChat class"""
    try:
        chat = DocumentChat()
        
        # Test document search and response generation
        query = "What are the contributions of Indian mathematicians to the field of astronomy?"
        response, usage_metadata = chat.generate_response(query)
        logging.info(f"Test response: {response}")
        logging.info(f"Usage metadata: {usage_metadata}")
        
        # Test aggregation queries
        test_queries = [
            "How many papers do you have?",
            "List all papers in your collection",
            "How many authors are there?",
            "What papers do you have by author Aryabhata?",
            "Summarize the collection"
        ]
        
        for query in test_queries:
            logging.info(f"\nTesting query: {query}")
            response = chat.perform_aggregation(query)
            logging.info(f"Response: {response}")
        
        # Test system prompts
        for prompt_name in ["default", "conversational", "expert"]:
            logging.info(f"\nTesting with prompt: {prompt_name}")
            chat = DocumentChat(prompt_name=prompt_name)
            response, _ = chat.generate_response("Tell me about Indian astronomy")
            logging.info(f"Response with {prompt_name} prompt (truncated):")
            logging.info(response["answer"][:100] + "...")
            
        # Test composite queries with prompt splitter
        logging.info("\n===== Testing Composite Queries with Prompt Splitter =====")
        ui = NavdhaniUI()
        
        composite_test_queries = [
            "Hello! I'm interested in Indian mathematics. How many papers do you have on this topic? Could you summarize the key concepts in Indian mathematics according to these papers?",
            
            "Hi, can you help me with two things? First, list all authors who wrote about astronomy. Second, what were the contributions of Aryabhata to mathematics? Please format the response in markdown.",
            
            "1. How many total papers are in the collection? 2. What are the major mathematical achievements described in these papers? 3. Display the results in a table format if possible.",
            
            "Thanks for your help earlier. Now I want information about the Kerala school of mathematics. Also, show me statistics about how many papers cover this topic. Format the output with bullet points.",
            
            "Hello. I have multiple questions: What was the concept of zero in Indian mathematics? How many papers discuss this? Who were the key contributors to this concept? Please make the response concise and well-structured."
        ]
        
        # Test each composite query
        for i, query in enumerate(composite_test_queries):
            logging.info(f"\nTesting composite query #{i+1}: {query}")
            
            # Use the prompt splitter to break down the query
            grouped_queries = ui.prompt_splitter.get_grouped_queries(query)
            
            # Log how the query was split
            logging.info(f"Query was split into {sum(len(queries) for queries in grouped_queries.values())} parts:")
            for query_type, queries in grouped_queries.items():
                if queries:
                    logging.info(f"- {query_type.name}: {len(queries)}")
                    for q in queries:
                        logging.info(f"  * {q}")
            
            # Process the composite query through NavdhaniUI
            chat_history = []
            history, stats, prompt_display = ui.respond(query, chat_history)
            
            # Log the response
            if history:
                logging.info(f"Response (truncated): {history[-1][1][:200]}...")
            logging.info(f"Stats: {stats}")
            
    except Exception as e:
        logging.error(f"Test execution failed: {e}")
        raise

if __name__ == "__main__":
    # Interactive mode detection with proper error handling
    INTERACTIVE_MODE = False
    try:
        from IPython.core.getipython import get_ipython
        shell = get_ipython().__class__.__name__
        INTERACTIVE_MODE = shell in ['ZMQInteractiveShell', 'TerminalInteractiveShell']
    except (NameError, ImportError):
        pass

    logging.info(f"Interactive mode: {INTERACTIVE_MODE}")
    try:
        if (INTERACTIVE_MODE):
            run_tests()
        else:
            pass

        # Initialize UI with default prompt
        ui = NavdhaniUI()
        logging.info("Starting Gradio interface...")
        start_time = time.time()
        ui.launch()
        # Note: Code after launch() won't execute until the Gradio server is shut down
        elapsed_time = time.time() - start_time
        logging.info(f"Gradio interface took {elapsed_time:.2f} seconds to launch")
    except Exception as e:
        logging.error(f"Application startup failed: {e}")
        raise

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
