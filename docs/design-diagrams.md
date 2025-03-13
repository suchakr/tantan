# Navadhāni System Architecture

This file contains detailed architecture diagrams for the Navadhāni document chat system.

## Core System Flow

```mermaid
flowchart TD
    User([User]) --> |"Query"| UI[NavdhaniUI]
    
    subgraph "Query Processing"
        UI --> |"Raw query"| PS[Prompt Splitter]
        PS --> |"Split queries"| QC[Query Classifier]
        QC --> |"Document query"| DC[Document Search]
        QC --> |"Aggregation query"| AH[Aggregation Handler]
        QC --> |"Conversational query"| DQ[Direct Query]
    end
    
    subgraph "Document Processing"
        DF[(Document TSV)] --> PP[Preprocessing]
        PP --> |"Raw text"| TC[Text Chunker]
        TC --> |"Text chunks"| TV[TF-IDF Vectorizer]
        TV --> |"Document vectors"| FAISS[FAISS Index]
        DC <--> |"Search"| FAISS
    end
    
    subgraph "Response Generation"
        DC --> |"Relevant documents"| LLM[Gemini LLM]
        AH --> |"Collection stats"| LLM
        DQ --> |"Direct prompt"| LLM
        LLM --> |"Raw response"| RF[Response Formatter]
        RF --> |"Structured response"| UI
    end
    
    UI --> |"Formatted response"| User
    
    classDef processing fill:#f9f,stroke:#333,stroke-width:2px
    classDef data fill:#bbf,stroke:#33a,stroke-width:2px
    classDef external fill:#bfb,stroke:#363,stroke-width:2px
    
    class User,LLM external
    class DF,FAISS data
    class PS,QC,DC,AH,DQ,PP,TC,TV,RF processing
```

## Component Relationships

```mermaid
classDiagram
    class DocumentChat {
        +chunk_size: int
        +chunk_overlap: int
        +df: DataFrame
        +chunks_df: DataFrame
        +vectorizer: TfidfVectorizer
        +document_embeddings: ndarray
        +index: faiss.IndexFlatL2
        +query_type: str
        +model: GenerativeModel
        +agg_handler: LLMAggregationHandler
        +create_chunks(text: str): List[str]
        +load_and_process_data(): void
        +search_documents(query: str, k: int): List[Dict]
        +is_aggregation_query(query: str): bool
        +is_conversational_query(query: str): bool
        +needs_document_context(query: str): bool
        +perform_aggregation(query: str): Dict
        +perform_aggregation_naive(query: str): Dict
        +generate_response(query: str): Tuple[Dict, Any]
    }
    
    class NavdhaniUI {
        +chat: DocumentChat
        +history: List[Tuple[str, str]]
        +token_stats: Dict[str, int]
        +last_prompt: str
        +context_used: bool
        +query_type: str
        +prompt_name: str
        +message_history: List[str]
        +history_index: int
        +example_queries: List[str]
        +prompt_splitter: PromptSplitter
        +update_stats(usage_metadata): str
        +respond(message: str, chat_history): Tuple
        +update_stats_from_multiple(usage_metadata_list): str
        +format_prompt_display(message: str, grouped_queries): str
        +reset_conversation(): Tuple
        +change_prompt(prompt_name: str): Tuple
        +use_example_query(example_query: str): Tuple
        +navigate_up(message: str): str
        +navigate_down(message: str): str
        +launch(share: bool): void
    }
    
    class DocumentChunk {
        +paper: str
        +author: str
        +url: str
        +text: str
        +original_idx: int
        +chunk_idx: int
    }
    
    class PromptSplitter {
        <<interface>>
        +get_grouped_queries(query: str): Dict[QueryType, List[str]]
    }
    
    class HybridPromptSplitter {
        +gemini_client: GenerativeModel
        +get_grouped_queries(query: str): Dict[QueryType, List[str]]
        +process_with_llm(query: str): Dict[QueryType, List[str]]
    }
    
    class HeuristicPromptSplitter {
        +get_grouped_queries(query: str): Dict[QueryType, List[str]]
        +split_by_delimiters(query: str): List[str]
    }
    
    class QueryType {
        <<enumeration>>
        DOCUMENT
        AGGREGATION
        CONVERSATIONAL
        FORMAT
        UNKNOWN
    }
    
    class LLMAggregationHandler {
        +dataset_path: str
        +df: DataFrame
        +process_query(query: str): Dict
    }
    
    NavdhaniUI --> DocumentChat : uses
    NavdhaniUI --> PromptSplitter : uses
    DocumentChat --> DocumentChunk : creates
    DocumentChat --> LLMAggregationHandler : uses
    HybridPromptSplitter --|> PromptSplitter : implements
    HeuristicPromptSplitter --|> PromptSplitter : implements
    PromptSplitter --> QueryType : uses
```

## Data Flow Sequence

### Query Processing Sequence

```mermaid
sequenceDiagram
    actor User
    participant UI as NavdhaniUI
    participant PS as PromptSplitter
    participant DC as DocumentChat
    participant VS as Vector Search
    participant AH as AggregationHandler
    participant LLM as Gemini LLM
    
    User->>UI: Submit query
    activate UI
    
    UI->>PS: Split composite query
    activate PS
    PS-->>UI: Grouped queries by type
    deactivate PS
    
    rect rgb(248, 250, 217)
    note right of UI: Query Processing Loop
    loop For each query by type
        UI->>DC: Process query
        activate DC
        
        alt Document query
            DC->>VS: Search for relevant documents
            activate VS
            VS-->>DC: Relevant document chunks
            deactivate VS
            
            DC->>LLM: Generate response with document context
            activate LLM
            LLM-->>DC: Structured response
            deactivate LLM
            
        else Aggregation query
            DC->>AH: Process aggregation query
            activate AH
            AH->>LLM: Generate statistical analysis
            activate LLM
            LLM-->>AH: Analysis results
            deactivate LLM
            AH-->>DC: Aggregation response
            deactivate AH
            
        else Conversational query
            DC->>LLM: Direct conversation without context
            activate LLM
            LLM-->>DC: Conversational response
            deactivate LLM
        end
        
        DC-->>UI: Query response
        deactivate DC
    end
    end
    
    UI->>UI: Format combined response
    UI->>UI: Update token statistics
    UI->>UI: Update conversation history
    
    UI-->>User: Display formatted response
    deactivate UI
```

### Data Ingestion Sequence

```mermaid
sequenceDiagram
    participant Main as Main Application
    participant DC as DocumentChat
    participant DF as Document File (TSV)
    participant PD as Pandas
    participant NLTK as NLTK
    participant TFIDF as TF-IDF Vectorizer
    participant FAISS as FAISS Index
    participant AH as Aggregation Handler
    participant LLM as Gemini LLM API
    participant UI as NavdhaniUI

    Main->>DC: Initialize DocumentChat
    activate DC
    
    DC->>DF: Load document TSV
    activate DF
    DF-->>PD: Raw data
    activate PD
    PD-->>DC: DataFrame with documents
    deactivate PD
    deactivate DF

    DC->>DC: Preprocess text
    
    DC->>NLTK: Initialize tokenizers
    activate NLTK
    NLTK-->>DC: Tokenization ready
    deactivate NLTK
    
    DC->>DC: Create document chunks
    Note over DC: Split texts into chunks with overlap
    
    DC->>TFIDF: Initialize vectorizer
    activate TFIDF
    DC->>TFIDF: Fit on document chunks
    TFIDF->>DC: Return fitted vectorizer
    DC->>TFIDF: Transform chunks to vectors
    TFIDF-->>DC: Document embeddings
    deactivate TFIDF
    
    DC->>FAISS: Initialize index
    activate FAISS
    DC->>FAISS: Add document vectors
    FAISS-->>DC: Index ready
    deactivate FAISS
    
    DC->>LLM: Initialize Gemini API client
    activate LLM
    LLM-->>DC: API client ready
    deactivate LLM
    
    DC->>AH: Initialize aggregation handler
    activate AH
    AH-->>DC: Aggregation handling ready
    deactivate AH
    
    DC-->>Main: DocumentChat initialized
    deactivate DC
    
    Main->>UI: Initialize NavdhaniUI with DocumentChat
    activate UI
    UI->>UI: Setup Gradio interface
    UI-->>Main: UI ready
    deactivate UI
    
    Main->>UI: Launch UI
    activate UI
    UI-->>Main: Serving on local/public URL
    deactivate UI