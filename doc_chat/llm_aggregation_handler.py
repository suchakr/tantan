#%%
"""
LLM Aggregation Query Handler for IJHS Dataset

This module uses an LLM to generate and execute Python code for answering
aggregation queries on the IJHS papers dataset. It takes natural language
queries, translates them into pandas operations, and returns structured responses.

Usage:
    from llm_aggregation_handler import LLMAggregationHandler
    
    handler = LLMAggregationHandler()
    result = handler.process_query("How many papers are about astronomy?")
    print(result['answer'])
"""

import os
import json
import logging
from time import sleep
import traceback
from typing import Dict, Any, List, Union, Optional, cast
import re
import pandas as pd
import numpy as np
from textwrap import dedent
from dotenv import load_dotenv, find_dotenv

# Import Google Generative AI
import google.generativeai as genai

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class LLMAggregationHandler:
    """
    Handles aggregation queries using LLM to generate executable pandas code
    based on natural language questions about the IJHS dataset
    """
    
    def __init__(self, 
                 dataset_path: str = 'ijhs-astro-math-docs.tsv',
                 model_name: str = "gemini-1.5-pro"):
        """
        Initialize the aggregation handler with dataset path and LLM settings
        
        Args:
            dataset_path: Path to the TSV dataset file
            model_name: Name of the Gemini model to use
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.df = None
        self.column_info = None
        
        # Load environment variables and API key
        dotenv_path = find_dotenv()
        load_dotenv(dotenv_path)
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please set GEMINI_API_KEY in your .env file")
        
        # Configure the Google Generative AI with the API key
        # Update the import and configuration approach
        import google.generativeai as genai
        genai.configure(api_key=api_key) # type: ignore
        
        # Create the model with appropriate configuration
        self.model = genai.GenerativeModel( # type: ignore
            model_name=model_name, 
            generation_config=genai.types.GenerationConfig( # type: ignore
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192
            )
        )
        
        # Load dataset and initialize column info
        self._load_dataset()
        
    def _load_dataset(self) -> None:
        """Load the dataset and create column information for the LLM prompt"""
        try:
            # Load dataset - drop columns that won't be useful for aggregations
            self.df = pd.read_csv(self.dataset_path, sep='\t').drop(
                columns=['pdf', 'full_pdf_path', 'pdf_exists', 'has_text']
            )
            
            # Generate column information
            columns_info = []
            for col in self.df.columns:
                dtype = str(self.df[col].dtype)
                sample = str(self.df[col].iloc[0])
                if len(sample) > 100:  # Truncate long text samples
                    sample = sample[:100] + "..."
                
                # For numeric columns, provide range information
                range_info = ""
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    try:
                        min_val = self.df[col].min()
                        max_val = self.df[col].max()
                        range_info = f" [range: {min_val} to {max_val}]"
                    except:
                        pass
                        
                columns_info.append(f"{col} ({dtype}){range_info}: Example - {sample}")
                
            self.column_info = "\n".join(columns_info)
            logging.info(f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            raise
    
    def generate_code(self, query: str) -> str:
        """
        Generate pandas code to answer the query using the LLM
        
        Args:
            query: Natural language query about the dataset
            
        Returns:
            Python code as a string that will produce the answer
        """
        prompt = dedent(f'''
        You are an expert data scientist who translates natural language questions into Python pandas code.
        
        Dataset Information:
        - The dataset is loaded into a pandas DataFrame called 'df'
        - The columns and their types are as follows:
        {self.column_info}
        
        Question: {query}
        
        Generate Python code using pandas to answer this question.
        For string types (object), you can use methods like 'contains', 'startswith', 'endswith', etc. in a case-insensitive manner. 
        Focus only on aggregation operations (count, sum, mean, group by, etc.). The code should:
        
        1. Be concise and efficient
        2. Return a clear, readable result (either a value, Series, or DataFrame)
        3. Include comments to explain what the code is doing
        4. Format the output nicely for human readability
        5. NOT use any external libraries other than pandas and its dependencies
        6. NOT include print statements or unnecessary assignments
        7. Be contained within a function called 'execute_query(df)'
        
        Only output valid Python code wrapped in ```python and ``` tags.
        Do not include any explanations outside the code block.
        ''')
        
        try:
            response = self.model.generate_content(prompt)
            code_text = response.text
            self.usage_metadata = response.usage_metadata
            
            # Extract code from markdown code block
            code_match = re.search(r'```python(.*?)```', code_text, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                return code
            else:
                # If not properly formatted with ```python ```, try to extract any code
                if '```' in code_text:
                    code = re.search(r'```(.*?)```', code_text, re.DOTALL)
                    if code:
                        return code.group(1).strip()
                
                # If still no code found, return the whole response
                return code_text.strip()
                
        except Exception as e:
            logging.error(f"Error generating code: {str(e)}")
            return f"# Error generating code: {str(e)}\n\ndef execute_query(df):\n    return 'Error generating code'"
    
    def execute_generated_code(self, code: str) -> Any:
        """
        Execute the generated pandas code and return the result
        
        Args:
            code: Python code to execute
            
        Returns:
            The result of executing the code
        """
        try:
            # Create a namespace for execution
            namespace = {'df': self.df, 'pd': pd, 'np': np}
            
            # Add the execute_query function to the namespace
            exec(code, namespace)
            
            # Now call the function with our dataframe
            if 'execute_query' in namespace:
                result = namespace['execute_query'](self.df)
                return result
            else:
                logging.error("No execute_query function found in generated code")
                return "Error: Generated code does not contain the required function"
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            logging.error(f"Error executing code: {error_traceback}")
            return f"Error executing code: {str(e)}"
    
    def format_result(self, result: Any) -> str:
        """
        Format the result of the code execution into a readable string
        
        Args:
            result: Result from code execution
            
        Returns:
            Formatted string representation of the result
        """
        try:
            if isinstance(result, pd.DataFrame):
                # For DataFrames, convert to a nicely formatted string
                if len(result) > 20:
                    # Show only first and last rows if DataFrame is large
                    return (
                        f"Total rows: {len(result)}\n\n"
                        f"First 10 rows:\n{result.head(10).to_string()}\n\n"
                        f"Last 10 rows:\n{result.tail(10).to_string()}"
                    )
                else:
                    return result.to_string()
                    
            elif isinstance(result, pd.Series):
                return result.to_string()
                
            elif isinstance(result, (list, tuple)):
                if len(result) > 30:
                    # Truncate long lists
                    items_str = ',\n'.join(str(item) for item in result[:15])
                    return f"List with {len(result)} items. First 15 items:\n{items_str}\n..."
                else:
                    return '\n'.join(str(item) for item in result)
                    
            elif isinstance(result, dict):
                return json.dumps(result, indent=2)
                
            else:
                return str(result)
                
        except Exception as e:
            logging.error(f"Error formatting result: {str(e)}")
            return f"Error formatting result: {str(e)}"

    def extract_references(self, result: Any) -> List[Dict[str, str]]:
        """
        Extract paper references from the result for inclusion in the response
        
        Args:
            result: Result from code execution
            
        Returns:
            List of paper reference dictionaries
        """
        references = []
        
        try:
            if isinstance(result, pd.DataFrame) and 'paper' in result.columns:
                # Take up to 5 papers from the result as references
                sample_rows = result.head(5)
                for _, row in sample_rows.iterrows():
                    ref = {
                        "title": row['paper'] if pd.notna(row['paper']) else "",
                        "author": row['author'] if 'author' in row and pd.notna(row['author']) else "",
                        "url": row['url'] if 'url' in row and pd.notna(row['url']) else ""
                    }
                    references.append(ref)
            
            # # If no references extracted from result, add some sample papers
            # if not references and isinstance(self.df, pd.DataFrame):
            #     sample_papers = self.df.drop_duplicates('paper').sample(min(3, len(self.df)))
            #     for _, row in sample_papers.iterrows():
            #         ref = {
            #             "title": '$$ ' + row['paper'],
            #             "author": row['author'] if pd.notna(row['author']) else "",
            #             "url": row['url'] if pd.notna(row['url']) else ""
            #         }
            #         references.append(ref)
                    
        except Exception as e:
            logging.warning(f"Error extracting references: {str(e)}")
            
        return references

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query about the dataset
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary containing answer, references, and confidence level
        """
        if not query.strip():
            return {
                "answer": "Empty query received. Please provide a question about the IJHS papers collection.",
                "references": [],
                "confidence_level": "high"
            }
            
        try:
            # Generate code
            generated_code = self.generate_code(query)
            usage_metadata = self.usage_metadata
            
            # Execute the code
            result = self.execute_generated_code(generated_code)
            # display("Result =", result)
            
            # Format the result
            formatted_result = self.format_result(result)
            
            # Extract references
            references = self.extract_references(result)
            
            # Determine confidence level based on execution success
            confidence_level = "high" if not isinstance(result, str) or not result.startswith("Error") else "low"
            
            # Build the final response
            response = {
                "answer": formatted_result,
                "references": references,
                "confidence_level": confidence_level,
                "generated_code": generated_code,  # Include the code for 
                "usage_metadata": usage_metadata
            }
            
            return response
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logging.error(f"Error processing query: {error_traceback}")
            
            return {
                "answer": f"Error processing query: {str(e)}",
                "references": [],
                "confidence_level": "low",
                "generated_code": "",
                "usage_metadata": {}
            }

    def cleanup(self):
        """Cleanup resources to ensure proper shutdown"""
        try:
            # Force cleanup of gRPC resources by closing the channel
            genai._client._client.transport.channel.close() # type: ignore
        except:
            pass

    def test_handler(self):
        """Run some test queries to verify functionality"""
        # handler = LLMAggregationHandler()
        
        try:
            test_queries = [
                "How many papers authored by Iyengar?" ,
                # "Which paper has the largest file size?",
                # "How many papers are in the collection?",
                # "Which subjects are covered in the papers and how many papers are there in each subject?",
                # "Who are the top 5 authors with the most papers?",
                # "What's the average size of papers in kilobytes?",
                # "How many papers are there about astronomy versus mathematics?",
            ]
            
            for query in test_queries[:]:
                print(f"\n\nTesting query: {query}")
                result = self.process_query(query)
                print(f"Answer: {result['answer']}")
                print("\nReferences:")
                for ref in result['references']:
                    print(f"\n----\n{ref['title']}")
                    if ref['author']: print(f"  Author: {ref['author']}")
                    if ref['url']: print(f"  URL: {ref['url']}")
                # print(f"Generated code:\n{result['generated_code']}")
                print(f"Confidence: {result['confidence_level']}")
                print(f"Usage metadata: {result['usage_metadata']}")
                sleep(2)  # To avoid rate limiting
        finally:
            # Ensure cleanup happens even if there's an exception
            # self.cleanup()
            pass

if __name__ == "__main__":
    small_file = "ijhs-astro-math-docs.tsv"
    big_file = "/Users/sunder/projects/cahc/cahc-utils/scrape/scraped/ijhs-classified-gemini_classify_text-textified~.tsv"
    handler = LLMAggregationHandler(dataset_path=big_file)
    handler.test_handler()
    handler.cleanup()

# %%
