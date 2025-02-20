#%%
import os
import pandas as pd
import numpy as np
import faiss
import gradio as gr
import google.generativeai as genai
from google.generativeai import GenerationConfig
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import cast
from scipy.sparse import spmatrix


import logging
# emit timestamp in logs
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
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-pro')

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

SYSTEM_PROMPT = """You are Navadhāni, an AI assistant specialized in discussing academic papers from the Indian Journal of History of Science (IJHS).

Your role is to:
1. Provide accurate information based on the academic papers you're given
2. Explain complex concepts in a clear and accessible way
3. Highlight important contributions from Indian scholars in mathematics and astronomy
4. Maintain academic integrity by staying true to the source material
5. Acknowledge when information is not available in the provided papers

Always base your responses on the context provided from the papers. If asked about topics outside the scope of the given papers, politely explain that you can only discuss content from the IJHS papers in your context."""


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
        self.load_and_process_data()

    def load_and_process_data(self):
        try:
            # Load TSV file
            self.df = pd.read_csv('ijhs-astro-math-docs.tsv', sep='\t').drop(columns=[ 'size_in_kb', 'cum_size_in_kb' , 'pdf', 'full_pdf_path', 'pdf_exists', 'has_text'])

            # print(f"Loaded {len(self.df)} documents. Shape: {self.df.shape}")
            
            # Initialize and fit TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer()
            self.document_embeddings = self.vectorizer.fit_transform(self.df['text'].fillna('')).toarray()
            print(f"Document embeddings shape: {self.document_embeddings.shape}")
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(self.document_embeddings.shape[1])
            self.index.add(self.document_embeddings.astype('float32'))
        except FileNotFoundError:
            print("Warning: ijhs-astro-math-docs.tsv not found. Please ensure the file exists in the current directory.")
            raise

    def search_documents(self, query, k=5):
        # Transform query to vector
        # query_vector = cast(spmatrix,cast(TfidfVectorizer, self.vectorizer).transform([query])).toarray().astype('float32')
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        
        # Search similar documents
        D, I = self.index.search(query_vector, k)
        return [self.df.iloc[i] for i in I[0]]

    def generate_response(self, query):
        try :
            # Get relevant documents
            relevant_docs = self.search_documents(query)
            # print(f"Found {len(relevant_docs)} relevant documents.")
            # print(relevant_docs)
    
            # Prepare context
            context = "\n".join([
                f"Title: {doc['paper']}\nAuthor: {doc['author']}\nContent: {doc['text'][:500]}..."
                for doc in relevant_docs
            ])
            context_json = str([ d.to_json() for d in relevant_docs])
    
            # Generate response using Gemini with system prompt
            prompt = f"""Based on the following academic papers from IJHS, answer this query: {query}

Context from relevant papers:
{context}

Please provide a clear and concise response based on the information from these papers."""
            # print(f"Prompt: {prompt}")

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

class NavdhaniUI:
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

if __name__ == "__main__" :
    ui = NavdhaniUI()
    ui.launch()
# %%
