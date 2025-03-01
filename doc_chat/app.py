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
from typing import cast, List, Dict
from scipy.sparse import spmatrix
import nltk
from nltk.tokenize import sent_tokenize

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
      "response_mime_type": "text/plain"
    #   "response_mime_type": "application/json",
    #   "response_schema": PaperClassifications
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

    def generate_response(self, query):
        try:
            # Get relevant documents
            relevant_chunks = self.search_documents(query)
            
            # Prepare context with chunked content
            context = "\n".join([
                f"Title: {chunk['paper']}\nAuthor: {chunk['author']}\nRelevant Extract: {chunk['text']}"
                for chunk in relevant_chunks
            ])
    
            # Generate response using Gemini with system prompt
            prompt = f"""Based on the following relevant extracts from IJHS papers, answer this query: {query}

Context from relevant papers:
{context}

Please provide a clear and concise response based on the information from these papers."""

            response = model.generate_content(prompt)
            return response.text, getattr(response, 'usage_metadata', None)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"Error: {str(e)}", None
        
#%%
# Run this block to test the DocumentChat class and its methods
def test_DocumentChat():
    chat = DocumentChat()
    query = "What are the contributions of Indian mathematicians to the field of astronomy?"
    # print(chat.search_documents(query))
    response, usage_metadata = chat.generate_response(query)
    print(response)
    print(usage_metadata)

if gInteractive:
    test_DocumentChat()


#%%
class NavdhaniUIPrev:
    def __init__(self):
        self.chat = DocumentChat()
        self.history = []
        self.token_stats = {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0}
        self.last_prompt = ""

    def update_stats(self, usage_metadata):
        if usage_metadata:
            try:
                # Update token statistics
                prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
                total_tokens = getattr(usage_metadata, 'total_token_count', 0)
                
                self.token_stats["prompt_tokens"] += prompt_tokens
                self.token_stats["response_tokens"] += completion_tokens
                self.token_stats["total_tokens"] += total_tokens

            except Exception as e:
                print(f"Error updating stats: {str(e)}")
        
        return f"""Token Usage Statistics:
Total Tokens: {self.token_stats['total_tokens']}
Prompt Tokens: {self.token_stats['prompt_tokens']}
Response Tokens: {self.token_stats['response_tokens']}"""

    def respond(self, message, history):
        if not history:
            history = []
        
        response, usage_metadata = self.chat.generate_response(message)
        stats = self.update_stats(usage_metadata)
        history.append((message, response))
        
        # Update last prompt with the full context
        relevant_docs = self.chat.search_documents(message)
        context = "\n".join([
            f"Title: {doc['paper']}\nAuthor: {doc['author']}\nContent: {doc['text'][:500]}..."
            for doc in relevant_docs
        ])
        self.last_prompt = f"""Based on the following academic papers from IJHS, answer this query: {message}

Context from relevant papers:
{context}

Please provide a clear and concise response based on the information from these papers."""
        
        return history, stats, self.last_prompt

    def launch(self):
        with gr.Blocks() as interface:
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=600)
                    msg = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message here...",
                    )
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

            msg.submit(
                self.respond,
                [msg, chatbot],
                [chatbot, stats_display, prompt_display]
            )

        interface.launch()

#%%
class NavdhaniUI:
    def __init__(self):
        self.chat = DocumentChat()
        self.history = []  # Maintain conversation history here
        self.token_stats = {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0}
        self.last_prompt = ""

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

        return f"""Token Usage Statistics:
Total Tokens: {self.token_stats['total_tokens']}
Prompt Tokens: {self.token_stats['prompt_tokens']}
Response Tokens: {self.token_stats['response_tokens']}"""

    def respond(self, message, chat_history):
        # Generate response and update internal history
        response, usage_metadata = self.chat.generate_response(message)
        stats = self.update_stats(usage_metadata)
        
        # Update self.history instead of a local variable
        self.history.append((message, response))
        
        # Update last prompt with the full context
        relevant_docs = self.chat.search_documents(message)
        context = "\n".join([
            f"Title: {doc['paper']}\nAuthor: {doc['author']}\nContent: {doc['text'][:500]}..."
            for doc in relevant_docs
        ])
        self.last_prompt = f"""Based on the following academic papers from IJHS, answer this query: {message}

Context from relevant papers:
{context}

Please provide a clear and concise response based on the information from these papers."""
        
        return self.history, stats, self.last_prompt

    def reset_conversation(self):
        # Reset conversation state
        self.history = []
        self.token_stats = {"total_tokens": 0, "prompt_tokens": 0, "response_tokens": 0}
        self.last_prompt = ""
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

#%%

if __name__ == "__main__" :
    ui = NavdhaniUI()
    ui.launch()
# %%
