import os
import streamlit as st
import google.generativeai as genai
import pandas as pd
import tiktoken
from dotenv import load_dotenv

# Must be the first Streamlit command
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="Papers Chat"
)

# Add CSS to create proper scrollable layout
st.markdown("""
<style>
    /* Core container reset */
    .stApp {
        overflow: hidden;
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0 !important;
    }

    /* Make chat container scrollable */
    .stChatMessageContainer {
        overflow-y: auto !important;
        max-height: calc(100vh - 200px) !important;
        margin-bottom: 100px !important;
    }

    /* Chat input styling */
    section:has(>.stChatInputContainer) {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: white !important;
        padding: 1rem 2rem !important;
        z-index: 1000 !important;
        border-top: 1px solid #ddd !important;
    }

    /* Ensure content doesn't get hidden */
    .element-container:has(> .stChatMessage) {
        margin-bottom: 0.5rem !important;
    }

    /* Column layout */
    [data-testid="column"] {
        padding: 0.5rem !important;
    }

    /* Hide unnecessary elements */
    #MainMenu {display: none;}
    footer {display: none;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Reset all containers */
    .main > .block-container {
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* App container */
    .stApp {
        position: absolute !important;
        left: 0 !important;
        right: 0 !important;
        top: 0 !important;
        bottom: 0 !important;
        overflow: hidden !important;
    }

    /* Message container */
    div[data-testid="stChatMessageContainer"] {
        position: absolute !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 100px !important;
        overflow-y: auto !important;
        padding: 1rem !important;
    }

    /* Input container */
    div[data-testid="stChatInputContainer"] {
        position: absolute !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        height: 100px !important;
        background: white !important;
        border-top: 1px solid #ddd !important;
        padding: 1rem !important;
        z-index: 999999 !important;
    }

    /* Hide scroll on main container */
    iframe[title="app"] {
        height: 100vh !important;
        overflow: hidden !important;
    }

    /* Hide default elements */
    #MainMenu, footer {display: none !important;}

    /* Stats column specific */
    [data-testid="column"]:first-child {
        border-right: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini and load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize model
model = genai.GenerativeModel('gemini-pro')

# Initialize session state
if 'total_tokens' not in st.session_state:
    st.session_state.total_tokens = {'user_input': 0, 'full_prompt': 0, 'response': 0, 'current_query': None}
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_full_prompt' not in st.session_state:
    st.session_state.last_full_prompt = ""
if 'gemini_tokens' not in st.session_state:
    st.session_state.gemini_tokens = {'prompt': 0, 'response': 0, 'total': 0, 'current_query': None}

# Optimized token counter using cached encoder
@st.cache_resource
def get_encoder():
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    encoder = get_encoder()
    return len(encoder.encode(text))

@st.cache_data
def load_papers():
    return pd.read_csv('ijhs-astro-math-docs.tsv', sep='\t')

def get_relevant_context(df, query, max_chars=10000):
    """Get relevant papers context based on query relevance score"""
    # Normalize query
    query_terms = set(term.lower() for term in query.split())
    
    # Score papers based on term frequency and position
    scores = []
    for idx, row in df.iterrows():
        text = str(row['text']).lower()
        
        # Calculate term frequency score
        term_count = sum(text.count(term) for term in query_terms)
        
        # Calculate position score (terms appearing earlier get higher weight)
        position_scores = []
        for term in query_terms:
            pos = text.find(term)
            if pos != -1:
                position_scores.append(1.0 / (1 + pos/100))
        position_score = sum(position_scores) if position_scores else 0
        
        # Combined score with position weight
        total_score = term_count + (position_score * 2)
        if total_score > 0:
            scores.append((total_score, idx))
    
    # Sort by relevance score
    scores.sort(reverse=True)
    
    # Build context from most relevant papers
    context = []
    total_chars = 0
    
    for score, idx in scores:
        paper_text = df.iloc[idx]['text']
        paper_title = df.iloc[idx]['paper']
        
        if total_chars + len(paper_text) <= max_chars:
            context.append(f"Paper: {paper_title}\n{paper_text}")
            total_chars += len(paper_text)
        else:
            # If full paper won't fit, include most relevant excerpt
            excerpt_len = max_chars - total_chars
            if excerpt_len > 200:  # Only include if we can get meaningful context
                # Find a good breakpoint near the end of the excerpt
                breakpoint = paper_text[:excerpt_len].rfind('. ') + 1
                if breakpoint > 0:
                    excerpt = paper_text[:breakpoint]
                    context.append(f"Paper: {paper_title} (excerpt)\n{excerpt}")
            break
    
    return "\n\n---\n\n".join(context) if context else "No relevant context found."

# Create main layout with fixed columns
col1, col2 = st.columns([1, 3], gap="medium")

# Stats column
with col1:
    st.header("Token Stats")
    
    # Current query metrics
    st.subheader("Current Query")
    current_cols = st.columns(3)
    with current_cols[0]:
        st.metric("Input", st.session_state.total_tokens['user_input'], help="Tokens in user query")
    with current_cols[1]:
        st.metric("Context", st.session_state.total_tokens['full_prompt'], help="Tokens with context")
    with current_cols[2]:
        st.metric("Response", st.session_state.total_tokens['response'], help="Tokens in response")
    
    st.markdown("---")
    
    # API metrics
    st.subheader("API Metrics")
    api_cols = st.columns(3)
    with api_cols[0]:
        st.metric("API Input", st.session_state.gemini_tokens['prompt'], help="API reported input tokens")
    with api_cols[1]:
        st.metric("API Output", st.session_state.gemini_tokens['response'], help="API reported output tokens")
    with api_cols[2]:
        st.metric("API Total", st.session_state.gemini_tokens['total'], help="Total API tokens")
    
    # Reset button
    if st.button("Reset Counts", use_container_width=True):
        st.session_state.total_tokens = {
            'user_input': 0, 
            'full_prompt': 0, 
            'response': 0, 
            'current_query': None
        }
        st.session_state.gemini_tokens = {
            'prompt': 0, 
            'response': 0, 
            'total': 0, 
            'current_query': None
        }
        st.rerun()
    
    # Debug expander
    with st.expander("View Full Prompt"):
        st.text_area(
            "Context + Query:",
            value=st.session_state.last_full_prompt,
            height=300,
            disabled=True
        )

# Chat column
with col2:
    st.header("Papers Chat")
    
    # Load papers data
    papers_df = load_papers()
    
    # Create a container for chat messages
    chat_container = st.container()
    
    # Create a container for input that will be fixed at bottom
    input_container = st.container()
    
    # Handle input first (so it appears at bottom)
    with input_container:
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        prompt = st.chat_input("Ask about the papers")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display messages in the chat container
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Handle new messages
    if prompt:
        # Reset counts for new query
        if st.session_state.total_tokens['current_query'] != prompt:
            st.session_state.total_tokens.update({
                'user_input': count_tokens(prompt),
                'full_prompt': 0,
                'response': 0,
                'current_query': prompt
            })
            st.session_state.gemini_tokens.update({
                'prompt': 0, 
                'response': 0, 
                'total': 0, 
                'current_query': prompt
            })
        
        # Add user message and display immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get context and prepare full prompt
        papers_context = get_relevant_context(papers_df, prompt)
        full_prompt = f"Context from relevant papers:\n{papers_context}\n\nUser question: {prompt}\n\nAnswer:"
        st.session_state.last_full_prompt = full_prompt
        st.session_state.total_tokens['full_prompt'] = count_tokens(full_prompt)
        
        # Generate and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                response = model.generate_content(full_prompt)
                message_placeholder.markdown(response.text)
                
                # Update token counts
                st.session_state.total_tokens['response'] = count_tokens(response.text)
                if hasattr(response, 'usage_metadata'):
                    st.session_state.gemini_tokens.update({
                        'prompt': response.usage_metadata.prompt_token_count,
                        'response': response.usage_metadata.candidates_token_count,
                        'total': response.usage_metadata.total_token_count
                    })
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.text
                })
                
            except Exception as e:
                message_placeholder.error(f"Error generating response: {str(e)}")
                # Reset counts on error
                st.session_state.total_tokens['response'] = 0
                st.session_state.gemini_tokens.update({
                    'prompt': 0,
                    'response': 0,
                    'total': 0
                })

st.markdown("</div>", unsafe_allow_html=True)