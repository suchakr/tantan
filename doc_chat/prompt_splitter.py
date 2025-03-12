"""
Prompt Splitting Module

This module provides functionality to split complex user prompts into individual
queries of different types. It defines an abstract base class PromptSplitter and
concrete implementations using heuristic and LLM-based approaches.

The main purpose is to identify separate queries within a single user prompt 
and group them logically by type for more effective processing.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
import re
import logging
import nltk
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import json

# Import the query classifier module for query type classification
from query_classifier import QueryType, QueryClassifier, PatternBasedQueryClassifier, LLMQueryClassifier


@dataclass
class QueryItem:
    """Represents a single query extracted from a complex prompt"""
    text: str  # The actual query text
    type: QueryType  # The classified type of the query
    priority: int  # Processing priority (lower = higher priority)
    original_position: int  # Original position in the prompt (for maintaining partial ordering)


class PromptSplitter(ABC):
    """Abstract base class defining the interface for prompt splitters"""
    
    def __init__(self, query_classifier: Optional[QueryClassifier] = None):
        """
        Initialize the prompt splitter with a query classifier
        
        Args:
            query_classifier: The classifier to use for classifying individual queries.
                             If None, a PatternBasedQueryClassifier will be used.
        """
        self.query_classifier = query_classifier or PatternBasedQueryClassifier()
        
        # Initialize NLTK for sentence tokenization
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    @abstractmethod
    def split_prompt(self, prompt: str) -> List[QueryItem]:
        """
        Split a complex prompt into individual queries and classify them
        
        Args:
            prompt: The user's complex prompt
            
        Returns:
            List[QueryItem]: A list of QueryItem objects containing the split queries,
                          their types, priorities, and original positions
        """
        pass
    
    def get_grouped_queries(self, prompt: str) -> Dict[QueryType, List[str]]:
        """
        Split the prompt and group the resulting queries by type
        
        Args:
            prompt: The user's complex prompt
            
        Returns:
            Dict[QueryType, List[str]]: Queries grouped by their types
        """
        query_items = self.split_prompt(prompt)
        
        # Group by type while maintaining priority order within each group
        groups: Dict[QueryType, List[str]] = {
            QueryType.CONVERSATIONAL: [],
            QueryType.AGGREGATION: [],
            QueryType.DOCUMENT: [],
            QueryType.FORMAT: [],
            QueryType.UNKNOWN: []
        }
        
        # Sort first by type, then by priority
        query_items.sort(key=lambda item: (item.type.name, item.priority))
        
        for item in query_items:
            groups[item.type].append(item.text)
        
        return groups
    
    def get_ordered_queries(self, prompt: str) -> List[Tuple[QueryType, str]]:
        """
        Split the prompt and return queries in a logical processing order
        
        Args:
            prompt: The user's complex prompt
            
        Returns:
            List[Tuple[QueryType, str]]: Queries with their types in processing order
        """
        query_items = self.split_prompt(prompt)
        
        # Sort by priority first, then by original position for ties
        query_items.sort(key=lambda item: (item.priority, item.original_position))
        
        return [(item.type, item.text) for item in query_items]


class HeuristicPromptSplitter(PromptSplitter):
    """
    Splits prompts into individual queries using rule-based heuristics
    """
    
    def __init__(self, query_classifier: Optional[QueryClassifier] = None):
        """
        Initialize with optional query classifier
        
        Args:
            query_classifier: The classifier to use (defaults to PatternBasedQueryClassifier)
        """
        super().__init__(query_classifier)
        
        # Priority mapping for query types (lower = higher priority)
        self.type_priority_map = {
            QueryType.CONVERSATIONAL: 1,  # Handle greetings first
            QueryType.FORMAT: 2,          # Then formatting preferences
            QueryType.AGGREGATION: 3,     # Then aggregation queries
            QueryType.DOCUMENT: 4,        # Then detailed document queries
            QueryType.UNKNOWN: 5          # Unknown queries last
        }
        
        # Regular expressions to identify query boundaries
        self.query_separators = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Sentence boundaries followed by capital letter
            r'(?<=[.!?])\s*\n+\s*',      # Sentence boundaries followed by newlines
            r'\n+\s*[-•*]\s+',           # Bullet points
            r'\n+\s*\d+[.)]\s+',         # Numbered lists
            r'\s+and\s+(?=(please|can|could|would)\s)',  # Conjunctions with request words
            r'\s+(?:also|additionally|moreover|furthermore)\s+(?=(?:please|can|could|would)\s)',  # Certain adverbs with request words
            r'(?<=\?)\s+'                # Question mark followed by space
        ]
        
        # Combine all separators into one pattern
        self.combined_separator = '|'.join(self.query_separators)
    
    def _get_priority(self, query_type: QueryType, text: str) -> int:
        """
        Determine the priority for a query based on its type and content
        
        Args:
            query_type: The type of the query
            text: The query text
            
        Returns:
            int: Priority value (lower = higher priority)
        """
        base_priority = self.type_priority_map.get(query_type, 99)
        
        # Adjust priority based on content analysis
        # Example: Greetings should come before questions even within the same type
        if query_type == QueryType.CONVERSATIONAL:
            if re.search(r'\b(hi|hello|greetings|hey)\b', text.lower()):
                return base_priority - 0.5  # Slight boost for greetings
        
        return base_priority
    
    def split_prompt(self, prompt: str) -> List[QueryItem]:
        """
        Split a complex prompt into individual queries using heuristic rules
        
        Args:
            prompt: The user's complex prompt
            
        Returns:
            List[QueryItem]: The identified individual queries with metadata
        """
        if not prompt or prompt.isspace():
            return []
        
        # Initial sentence splitting
        sentences = sent_tokenize(prompt)
        
        # Further split by potential query separators
        raw_queries: List[str] = []
        position = 0
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
                
            # Check for explicit separators within the sentence
            segments = re.split(self.combined_separator, sentence)
            
            # Filter out None values and empty strings before stripping
            segments = [seg for seg in segments if seg]
            segments = [seg.strip() for seg in segments if seg.strip()]
            
            if len(segments) > 1:
                raw_queries.extend(segments)
            else:
                raw_queries.append(sentence)
        
        # Special handling for numbered/bulleted lists
        # Look for patterns like "1. item", "2. item" in the raw prompt
        numbered_items = re.findall(r'(?:^|\n)\s*(\d+[.)]\s+[^\n]+)', prompt)
        bulleted_items = re.findall(r'(?:^|\n)\s*([-•*]\s+[^\n]+)', prompt)
        list_items = numbered_items + bulleted_items
        
        # If we found list items and they're not already in raw_queries, add them
        for item in list_items:
            if item.strip() not in raw_queries:
                raw_queries.append(item.strip())
        
        # Process each raw query text to create QueryItems
        query_items: List[QueryItem] = []
        
        for i, raw_query in enumerate(raw_queries):
            # Skip very short or meaningless segments (likely punctuation or separators)
            if len(raw_query.split()) < 2 and not re.search(r'\?|!', raw_query):
                continue
                
            # Classify the query
            query_type = self.query_classifier.classify_query(raw_query)
            
            # Determine priority
            priority = self._get_priority(query_type, raw_query)
            
            # Create and add the QueryItem
            query_items.append(QueryItem(
                text=raw_query,
                type=query_type,
                priority=priority,
                original_position=i
            ))
        
        # Clean up and merge related queries if needed
        return self._clean_up_queries(query_items)
    
    def _clean_up_queries(self, query_items: List[QueryItem]) -> List[QueryItem]:
        """
        Clean up the list of query items by merging related consecutive items
        
        Args:
            query_items: The initial list of query items
            
        Returns:
            List[QueryItem]: The cleaned up list
        """
        if not query_items or len(query_items) <= 1:
            return query_items
            
        result: List[QueryItem] = []
        current_buffer: List[QueryItem] = [query_items[0]]
        
        for i in range(1, len(query_items)):
            current = query_items[i]
            previous = current_buffer[-1]
            
            # Check if the current query should be merged with the previous one
            # based on various conditions
            should_merge = (
                current.type == previous.type and  # Same type
                current.original_position == previous.original_position + 1 and  # Sequential
                len(previous.text) + len(current.text) < 200  # Not too long when combined
            )
            
            # Additional check: short fragments that don't make sense on their own
            if len(current.text.split()) < 3 and not re.search(r'\?|!', current.text):
                should_merge = True
                
            if should_merge:
                # Merge with the previous item
                current_buffer[-1] = QueryItem(
                    text=previous.text + " " + current.text,
                    type=previous.type,
                    priority=min(previous.priority, current.priority),
                    original_position=previous.original_position
                )
            else:
                # If we can't merge with the previous item, add all buffered items 
                # to the result and start a new buffer
                result.extend(current_buffer)
                current_buffer = [current]
        
        # Add any remaining buffer items
        result.extend(current_buffer)
        
        return result


class LLMPromptSplitter(PromptSplitter):
    """
    Splits prompts using a Language Model for more nuanced understanding
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        query_classifier: Optional[QueryClassifier] = None
    ):
        """
        Initialize the LLM-based prompt splitter
        
        Args:
            api_key: Google API key for Gemini (optional)
            query_classifier: The classifier to use for query classification
        """
        super().__init__(query_classifier)
        self._model = None
        self.api_key = api_key
        
        # System prompt for LLM-based splitting
        self.system_prompt = """
        You are a query analyzer for a document chat system. Your job is to identify separate
        queries within a complex user prompt and output them as a JSON array. Each query should include:
        
        1. "text": The full text of the identified query
        2. "type": One of: "DOCUMENT", "CONVERSATIONAL", "AGGREGATION", "FORMAT"
        3. "priority": A number from 1-10 indicating processing priority (lower = higher priority)
        
        Example output:
        [
          {
            "text": "Hello, I'm looking for information",
            "type": "CONVERSATIONAL", 
            "priority": 1
          },
          {
            "text": "Show me all papers by Aryabhata",
            "type": "AGGREGATION",
            "priority": 3
          }
        ]
        
        Rules for splitting:
        - Split on natural language boundaries and clear topic shifts
        - Group closely related queries of the same type together
        - Prioritize queries in this order: greetings > formatting requests > collection statistics > document-specific questions
        - Maintain logical flow between related queries
        - Ensure each query is complete and meaningful on its own
        
        Output ONLY the JSON array, no markdown formatting or backticks.
        """
        
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=5,
            max_output_tokens=4096,
        )
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Gemini model if API key is available"""
        try:
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=self.generation_config,
                    system_instruction=self.system_prompt,
                )
            else:
                logging.warning("No API key provided for LLMPromptSplitter")
        except Exception as e:
            logging.error(f"Failed to initialize LLM splitter model: {e}")
            self._model = None
    
    def _clean_json_response(self, text: str) -> str:
        """
        Clean the JSON response from LLM to make it parseable
        
        Args:
            text: The raw text response from the LLM
            
        Returns:
            str: Cleaned JSON string
        """
        # Remove markdown code blocks if present
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        # Remove any other markdown formatting
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        # Remove any explanatory text before or after the JSON
        text = re.sub(r'^.*?\[', '[', text, flags=re.DOTALL)
        text = re.sub(r'\].*?$', ']', text, flags=re.DOTALL)
        
        return text
    
    def split_prompt(self, prompt: str) -> List[QueryItem]:
        """
        Split a complex prompt using the LLM
        
        Args:
            prompt: The user's complex prompt
            
        Returns:
            List[QueryItem]: The identified individual queries with metadata
        """
        if not prompt or prompt.isspace():
            return []
            
        if not self._model:
            logging.warning("LLM not available, falling back to heuristic splitting")
            return HeuristicPromptSplitter(self.query_classifier).split_prompt(prompt)
        
        try:
            # Ask the LLM to split the prompt
            response = self._model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean the JSON response to make it parseable
            cleaned_json_text = self._clean_json_response(result_text)
            
            # Parse the JSON response
            try:
                parsed_result = json.loads(cleaned_json_text)
                
                # Validate and convert to QueryItems
                query_items: List[QueryItem] = []
                
                for i, item in enumerate(parsed_result):
                    if not isinstance(item, dict):
                        continue
                        
                    query_text = item.get("text", "").strip()
                    if not query_text:
                        continue
                        
                    # Get the query type
                    type_str = item.get("type", "UNKNOWN").upper()
                    query_type = QueryType.UNKNOWN
                    
                    # Map string types to QueryType enum
                    type_mapping = {
                        "DOCUMENT": QueryType.DOCUMENT,
                        "CONVERSATIONAL": QueryType.CONVERSATIONAL,
                        "AGGREGATION": QueryType.AGGREGATION,
                        "FORMAT": QueryType.FORMAT
                    }
                    query_type = type_mapping.get(type_str, QueryType.UNKNOWN)
                    
                    # Get priority
                    try:
                        priority = int(item.get("priority", 5))
                        priority = max(1, min(10, priority))  # Clamp to 1-10
                    except (ValueError, TypeError):
                        priority = 5  # Default priority
                    
                    query_items.append(QueryItem(
                        text=query_text,
                        type=query_type,
                        priority=priority,
                        original_position=i
                    ))
                
                return query_items
                
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing LLM response: {e}")
                logging.error(f"Raw response: {result_text}")
                logging.error(f"Cleaned response: {cleaned_json_text}")
                # Fall back to heuristic splitting
                return HeuristicPromptSplitter(self.query_classifier).split_prompt(prompt)
                
        except Exception as e:
            logging.error(f"Error in LLM prompt splitting: {e}")
            # Fall back to heuristic splitting
            return HeuristicPromptSplitter(self.query_classifier).split_prompt(prompt)


class HybridPromptSplitter(PromptSplitter):
    """
    A prompt splitter that combines both heuristic and LLM-based approaches
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        query_classifier: Optional[QueryClassifier] = None,
        complexity_threshold: int = 50  # Character count threshold
    ):
        """
        Initialize the hybrid prompt splitter
        
        Args:
            api_key: Google API key for Gemini (optional)
            query_classifier: The classifier to use for query classification
            complexity_threshold: Character count threshold for using LLM
        """
        super().__init__(query_classifier)
        self.heuristic_splitter = HeuristicPromptSplitter(query_classifier)
        self.llm_splitter = LLMPromptSplitter(api_key, query_classifier)
        self.complexity_threshold = complexity_threshold
    
    def split_prompt(self, prompt: str) -> List[QueryItem]:
        """
        Split a complex prompt using either heuristic or LLM-based approach
        based on prompt complexity
        
        Args:
            prompt: The user's complex prompt
            
        Returns:
            List[QueryItem]: The identified individual queries with metadata
        """
        if not prompt or prompt.isspace():
            return []
        
        # Evaluate prompt complexity based on length, structure, etc.
        # For simplicity, we'll start with just character count
        is_complex = len(prompt) > self.complexity_threshold
        
        # Additional complexity heuristics could include:
        # - Multiple question marks (indicating multiple questions)
        # - Multiple sentences with different subjects
        # - Presence of bullet points or numbered lists
        has_multiple_questions = prompt.count('?') > 1
        has_bullet_points = bool(re.search(r'[-•*]\s+\w+', prompt))
        has_numbered_list = bool(re.search(r'\d+[.)]\s+\w+', prompt))
        
        is_complex = is_complex or has_multiple_questions or has_bullet_points or has_numbered_list
        
        if is_complex and self.llm_splitter._model:
            # Use LLM for complex prompts if available
            return self.llm_splitter.split_prompt(prompt)
        else:
            # Fall back to heuristic for simple prompts or if LLM is unavailable
            return self.heuristic_splitter.split_prompt(prompt)


# Test functionality
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    
    # Create splitters for testing
    heuristic_splitter = HeuristicPromptSplitter()
    
    # Optional LLM splitter with API key from environment
    import os
    from dotenv import load_dotenv, find_dotenv
    
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    api_key = os.getenv('GEMINI_API_KEY')
    
    llm_splitter = None
    hybrid_splitter = None
    if api_key:
        llm_splitter = LLMPromptSplitter(api_key)
        hybrid_splitter = HybridPromptSplitter(api_key)
    
    # Test prompts
    test_prompts = [
        "Hello! Can you find papers by Aryabhata on astronomy and also give me a count of all documents in the collection?",
        
        "Hi there. I'd like to know how many papers are in the collection. Also, what were the major contributions of Indian mathematicians to trigonometry? Could you display the results in a table format?",
        
        "Show me all authors who published after 1950. What were the key discoveries in Indian astronomy during the medieval period according to these papers?",
        
        "1. List all papers on mathematics\n2. Who was the most prolific author?\n3. Summarize the key findings on astronomy",
        
        "Thanks for your help earlier. Now I want to know about the Kerala school of mathematics. How many papers do you have on this topic?"
    ]
    
    def display_split_results(prompt: str, splitter: PromptSplitter, splitter_name: str) -> None:
        """Display the results of splitting a prompt"""
        print(f"\n===== {splitter_name} =====")
        print(f"Original prompt: {prompt}")
        print("\nSplit queries:")
        
        try:
            query_items = splitter.split_prompt(prompt)
            for i, item in enumerate(query_items):
                print(f"{i+1}. [{item.type.name}] (Priority: {item.priority}) - {item.text}")
            
            print("\nGrouped by type:")
            grouped = splitter.get_grouped_queries(prompt)
            for query_type, queries in grouped.items():
                if queries:  # Only show non-empty groups
                    print(f"{query_type.name}:")
                    for query in queries:
                        print(f"  - {query}")
            
            print("\nIn logical processing order:")
            ordered = splitter.get_ordered_queries(prompt)
            for i, (query_type, query) in enumerate(ordered):
                print(f"{i+1}. [{query_type.name}] - {query}")
        except Exception as e:
            print(f"Error processing prompt: {e}")
        
        print("-" * 80)
    
    # Test each splitter with all prompts
    for prompt in test_prompts:
        # Always test the heuristic splitter
        display_split_results(prompt, heuristic_splitter, "Heuristic Prompt Splitter")
        
        # Test LLM and hybrid splitters if available
        if llm_splitter:
            display_split_results(prompt, llm_splitter, "LLM-Based Prompt Splitter")
            
        if hybrid_splitter:
            display_split_results(prompt, hybrid_splitter, "Hybrid Prompt Splitter")
    
    if not api_key:
        print("\nLLM-based and Hybrid splitters not available (API key not found)")