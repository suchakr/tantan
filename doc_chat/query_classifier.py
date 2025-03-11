"""
Query Classification Module

This module provides different strategies for classifying user queries in the document chat system.
It defines an abstract base class QueryClassifier and concrete implementations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Optional, Union, Tuple
import logging
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

class QueryType(Enum):
    """Enumeration of possible query types"""
    DOCUMENT = "document"  # Query requires searching through documents
    CONVERSATIONAL = "conversational"  # General conversational query
    AGGREGATION = "aggregation"  # Query about collection statistics
    FORMAT = "format"  # Query about formatting or display preferences
    UNKNOWN = "unknown"  # Unclassified query type

class QueryClassifier(ABC):
    """Abstract base class defining the interface for query classifiers"""

    @abstractmethod
    def classify_query(self, query: str) -> QueryType:
        """
        Classify the given query into one of the predefined query types
        
        Args:
            query: The user's query text
            
        Returns:
            QueryType: The classified type of the query
        """
        pass
    
    def is_conversational_query(self, query: str) -> bool:
        """
        Check if a query is conversational in nature
        
        Args:
            query: The user's query text
            
        Returns:
            bool: True if the query is conversational
        """
        return self.classify_query(query) == QueryType.CONVERSATIONAL
    
    def is_aggregation_query(self, query: str) -> bool:
        """
        Check if a query is requesting collection-level statistics
        
        Args:
            query: The user's query text
            
        Returns:
            bool: True if the query is requesting aggregation
        """
        return self.classify_query(query) == QueryType.AGGREGATION
    
    def is_document_query(self, query: str) -> bool:
        """
        Check if a query requires document context
        
        Args:
            query: The user's query text
            
        Returns:
            bool: True if the query requires document context
        """
        return self.classify_query(query) == QueryType.DOCUMENT
    
    def needs_document_context(self, query: str) -> bool:
        """
        Determine if a query requires document context for answering
        
        Args:
            query: The user's query text
            
        Returns:
            bool: True if the query requires document context
        """
        return self.is_document_query(query)


class PatternBasedQueryClassifier(QueryClassifier):
    """
    A query classifier that uses predefined patterns to classify queries
    This implements the current approach used in the application
    """
    
    def __init__(self):
        """Initialize pattern lists for different query types"""
        self.conversational_patterns: List[str] = [
            'who are you', 'your name', 'what can you do', 'hello', 'hi',
            'help me', 'how do you work', 'what is your purpose',
            'thanks', 'thank you'
        ]
        
        self.aggregation_patterns: List[str] = [
            'how many papers', 'how many documents', 'paper count',
            'list all papers', 'list papers', 'show papers',
            'what papers do you have', 'what documents are available',
            'summarize the papers', 'summarize the collection',
            'which authors', 'list authors', 'how many authors',
            'most cited', 'most popular', 'most common topics',
            'papers by year', 'papers by author', 'papers by topic',
            'statistics', 'overview', 'what years', 'date range'
        ]

    def is_conversational_query(self, query: str) -> bool:
        """
        Check if a query is conversational in nature
        
        Args:
            query: The user's query text
            
        Returns:
            bool: True if the query is conversational
        """
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in self.conversational_patterns
                  ) and not self.is_aggregation_query(query)

    def is_aggregation_query(self, query: str) -> bool:
        """
        Check if a query is requesting collection-level statistics
        
        Args:
            query: The user's query text
            
        Returns:
            bool: True if the query is requesting aggregation
        """
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in self.aggregation_patterns)
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify the given query based on pattern matching
        
        Args:
            query: The user's query text
            
        Returns:
            QueryType: The classified type of the query
        """
        query_lower = query.lower()
        
        # First check for aggregation queries (highest priority)
        if any(pattern in query_lower for pattern in self.aggregation_patterns):
            return QueryType.AGGREGATION
        
        # Then check for conversational queries
        if any(pattern in query_lower for pattern in self.conversational_patterns):
            return QueryType.CONVERSATIONAL
        
        # Default to document query if no other patterns match
        return QueryType.DOCUMENT


class LLMQueryClassifier(QueryClassifier):
    """
    A query classifier that uses a Language Model to classify queries
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM classifier
        
        Args:
            api_key: Google API key for Gemini (optional)
        """
        self._model = None
        self.api_key = api_key
        self.system_prompt = """
        You are a query classifier for a document chat system. Your job is to categorize user queries into one of the following types:
        
        1. DOCUMENT: Queries that require searching through specific documents for information
        2. CONVERSATIONAL: General conversational queries like greetings or questions about the system
        3. AGGREGATION: Queries about collection-level statistics or listings of documents
        4. FORMAT: Queries about formatting or display preferences
        
        Respond with ONLY the category name, nothing else.
        """
        
        self.generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            top_k=3,
            max_output_tokens=10,
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
                logging.warning("No API key provided for LLMQueryClassifier")
        except Exception as e:
            logging.error(f"Failed to initialize LLM classifier model: {e}")
            self._model = None
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify the query using the LLM
        
        Args:
            query: The user's query text
            
        Returns:
            QueryType: The classified type of the query
        """
        if not self._model:
            # Fall back to pattern-based classifier if model is unavailable
            logging.warning("LLM not available, falling back to pattern-based classification")
            return PatternBasedQueryClassifier().classify_query(query)
        
        try:
            response = self._model.generate_content(query)
            result_text = response.text.strip().upper()
            
            # Map the LLM response to QueryType enum
            type_mapping = {
                "DOCUMENT": QueryType.DOCUMENT,
                "CONVERSATIONAL": QueryType.CONVERSATIONAL,
                "AGGREGATION": QueryType.AGGREGATION,
                "FORMAT": QueryType.FORMAT
            }
            
            return type_mapping.get(result_text, QueryType.UNKNOWN)
        
        except Exception as e:
            logging.error(f"Error classifying query with LLM: {e}")
            # Fall back to pattern-based classification on error
            return PatternBasedQueryClassifier().classify_query(query)


class HybridQueryClassifier(QueryClassifier):
    """
    A classifier that combines pattern-based and LLM-based classification
    using the LLM for uncertain cases
    """
    
    def __init__(self, api_key: Optional[str] = None, confidence_threshold: float = 0.7):
        """
        Initialize the hybrid classifier
        
        Args:
            api_key: Google API key for Gemini (optional)
            confidence_threshold: Threshold for using pattern classifier
        """
        self.pattern_classifier = PatternBasedQueryClassifier()
        self.llm_classifier = LLMQueryClassifier(api_key)
        self.confidence_threshold = confidence_threshold
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify the query using both pattern and LLM classifiers
        
        First tries pattern classification, and if confidence is high enough
        returns that result. Otherwise falls back to LLM classification.
        
        Args:
            query: The user's query text
            
        Returns:
            QueryType: The classified type of the query
        """
        # First try pattern-based classification
        pattern_result = self.pattern_classifier.classify_query(query)
        
        # If pattern classifier returned DOCUMENT (default), we might want to use LLM
        if pattern_result == QueryType.DOCUMENT:
            # Use LLM for potentially ambiguous queries longer than 5 words
            if len(query.split()) > 5:
                return self.llm_classifier.classify_query(query)
        
        return pattern_result


# Test functionality
if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    
    # Create classifiers for testing
    pattern_classifier = PatternBasedQueryClassifier()
    
    # Optional LLM classifier with API key from environment
    import os
    from dotenv import load_dotenv, find_dotenv
    
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    api_key = os.getenv('GEMINI_API_KEY')
    
    llm_classifier = None
    hybrid_classifier = None
    if api_key:
        llm_classifier = LLMQueryClassifier(api_key)
        hybrid_classifier = HybridQueryClassifier(api_key)
    
    # Test queries with expected results
    test_queries_with_expected_result = [
        ["What are the contributions of Indian mathematicians to astronomy?", QueryType.DOCUMENT],
        ["How many papers do you have in your collection?", QueryType.AGGREGATION],
        ["Hello, what can you do?", QueryType.CONVERSATIONAL],
        ["List all papers by author Iyengar", QueryType.AGGREGATION],
        ["List all papers on Aryabhata", QueryType.DOCUMENT],
        ["What's the average length of papers in the collection?", QueryType.AGGREGATION],
        ["Can you display the results in a table format?", QueryType.FORMAT],
        ["Thanks for your help!", QueryType.CONVERSATIONAL],
        ["Who developed the concept of zero in Indian mathematics?", QueryType.DOCUMENT]
    ]
    
    # Function to evaluate accuracy
    def evaluate_classifier(classifier: QueryClassifier, test_data: List[List], classifier_name: str) -> Tuple[List[Dict], float]:
        results = []
        correct_count = 0
        
        print(f"\n===== Testing {classifier_name} =====")
        print(f"{'Query':<50} | {'Expected':<15} | {'Predicted':<15} | {'Correct':<10}")
        print("-" * 95)
        
        for query, expected_type in test_data:
            predicted_type = classifier.classify_query(query)
            is_correct = predicted_type == expected_type
            if is_correct:
                correct_count += 1
                
            result = {
                "query": query,
                "expected": expected_type,
                "predicted": predicted_type,
                "is_correct": is_correct
            }
            results.append(result)
            
            # Print the tabulated row
            query_truncated = query[:47] + "..." if len(query) > 50 else query.ljust(50)
            print(f"{query_truncated:<50} | {expected_type.name:<15} | {predicted_type.name:<15} | {'✓' if is_correct else '✗':<10}")
        
        accuracy = correct_count / len(test_data) if test_data else 0
        print("-" * 95)
        print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(test_data)})")
        
        return results, accuracy
    
    # Test all available classifiers
    pattern_results, pattern_accuracy = evaluate_classifier(
        pattern_classifier, 
        test_queries_with_expected_result, 
        "Pattern-Based Classifier"
    )
    
    llm_results, llm_accuracy = None, 0.0
    hybrid_results, hybrid_accuracy = None, 0.0
    
    if llm_classifier:
        llm_results, llm_accuracy = evaluate_classifier(
            llm_classifier, 
            test_queries_with_expected_result, 
            "LLM-Based Classifier"
        )
        
        hybrid_results, hybrid_accuracy = evaluate_classifier(
            hybrid_classifier, 
            test_queries_with_expected_result, 
            "Hybrid Classifier"
        )
        
        # Summary table
        print("\n===== Classification Accuracy Summary =====")
        print(f"{'Classifier':<25} | {'Accuracy':<15}")
        print("-" * 45)
        print(f"{'Pattern-Based Classifier':<25} | {pattern_accuracy:.2%}")
        print(f"{'LLM-Based Classifier':<25} | {llm_accuracy:.2%}")
        print(f"{'Hybrid Classifier':<25} | {hybrid_accuracy:.2%}")
    else:
        print("\nLLM-based and Hybrid classifiers not available (API key not found)")
        
        # Summary table for pattern only
        print("\n===== Classification Accuracy Summary =====")
        print(f"{'Classifier':<25} | {'Accuracy':<15}")
        print("-" * 45)
        print(f"{'Pattern-Based Classifier':<25} | {pattern_accuracy:.2%}")