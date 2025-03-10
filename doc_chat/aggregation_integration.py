"""
Example integration of LLMAggregationHandler with the DocumentChat application

This file shows how to integrate the LLMAggregationHandler with the main app.py
This is just for demonstration - you can integrate this code directly into app.py later.
"""

from llm_aggregation_handler import LLMAggregationHandler
from app import DocumentChat, NavdhaniUI

# Sample integration function showing how to use LLMAggregationHandler in the DocumentChat class
def integrated_generate_response(chat_instance, query: str):
    """
    Enhanced version of generate_response that uses LLMAggregationHandler for aggregation queries
    
    Args:
        chat_instance: An instance of the DocumentChat class
        query: The user's question
        
    Returns:
        The response and metadata, same format as the original generate_response
    """
    # Check if it's an aggregation query using the existing method
    if chat_instance.is_aggregation_query(query):
        # Instead of using the existing perform_aggregation method,
        # use the LLM aggregation handler
        try:
            handler = LLMAggregationHandler()
            response = handler.process_query(query)
            
            # Log the generated code for debugging
            print(f"Generated code for query '{query}':")
            print(response.get('generated_code', 'No code generated'))
            
            # Remove generated_code from response since it's just for debugging
            if 'generated_code' in response:
                del response['generated_code']
                
            return response, None  # Return response, no usage_metadata
            
        except Exception as e:
            print(f"Error with LLM aggregation handler: {str(e)}")
            # Fall back to the original method
            return chat_instance.perform_aggregation(query), None
    else:
        # Use the original method for non-aggregation queries
        return chat_instance.generate_response(query)

# Example of how to patch the DocumentChat class to use the new method
def patch_document_chat():
    """Patch the DocumentChat class to use LLMAggregationHandler"""
    original_generate_response = DocumentChat.generate_response
    
    def patched_generate_response(self, query: str):
        return integrated_generate_response(self, query)
    
    DocumentChat.generate_response = patched_generate_response
    print("DocumentChat class patched to use LLMAggregationHandler!")

# Example usage of the patch
if __name__ == "__main__":
    # This is how you would apply the patch before creating the UI
    patch_document_chat()
    
    # This would create the UI with the enhanced aggregation capability
    # ui = NavdhaniUI()
    # ui.launch()
    
    # For testing without launching the UI:
    chat = DocumentChat()
    test_queries = [
        "How many papers are in the collection?",
        "List all papers with their authors",
        "What subjects are covered in the papers?",
        "How many papers are about astronomy versus mathematics?",
        "Who's written the most papers in this collection?"
    ]
    
    for query in test_queries:
        print(f"\n\nTesting query: {query}")
        response, _ = chat.generate_response(query)
        print(f"Answer: {response['answer']}")
        print(f"References: {len(response['references'])} included")
        print(f"Confidence: {response['confidence_level']}")