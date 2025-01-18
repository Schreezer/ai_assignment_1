import logging
from typing import List, Tuple, Dict
from datasets import load_dataset
import google.generativeai as genai
import os
import time
from ratelimit import limits, sleep_and_retry
import pickle
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_api() -> genai.GenerativeModel:
    """Set up the Gemini model with error checking."""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("Missing Gemini API key in environment variables")
    
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash-thinking-exp-1219",
        generation_config=generation_config
    )
    
    return model

def query_format_helper(pronoun: str, answers: List[str]) -> str:
    """Format a query about pronoun reference given possible answers."""
    if not pronoun or not all(answers):
        raise ValueError("Pronoun and answers must not be empty")
    return f'Q: In the previous statement, does "{pronoun}" refer to {answers[0]} or {answers[1]}? A:'

def construct_query_from_schema(
    text: str, 
    pronoun: str, 
    answers: List[str]
) -> str:
    """Construct a complete query combining the text and formatted question."""
    if not text or not pronoun:
        raise ValueError("Text and pronoun cannot be empty")
    if len(answers) != 2:
        raise ValueError("Must provide exactly two possible answers")
    return f"S:{text} {query_format_helper(pronoun, answers)}"

def extract_answer(response: str, options: List[str]) -> str:
    """Extract the answer from the model's response and match it to the closest option."""
    response_lower = response.lower()
    options_lower = [opt.lower() for opt in options]
    
    # Try to find exact matches first
    for i, option in enumerate(options_lower):
        if option in response_lower:
            return options[i]
    
    # If no exact match, try fuzzy matching
    for i, option in enumerate(options_lower):
        words = option.split()
        if any(word in response_lower for word in words):
            return options[i]
    
    logger.warning(f"Could not match response '{response}' to any option in {options}")
    return options[0]

@sleep_and_retry
@limits(calls=6, period=60)  # 6 calls per 60 seconds
def get_gemini_answer(
    model: genai.GenerativeModel, 
    query_prompt: str
) -> str:
    """Get answer from Gemini API for a Winograd Schema query using chat history."""
    try:
        # Initialize chat with example conversation history
        chat = model.start_chat(history=[
            {
                "role": "user",
                "parts": ["Let's solve some Winograd Schema questions about pronoun references."]
            },
            {
                "role": "model",
                "parts": ["I'll help you resolve pronoun references in Winograd Schema questions. Please provide the statements and questions, and I'll identify what the pronouns refer to."]
            }
        ])
        
        logger.info(f"Sending query to Gemini API: {query_prompt}")
        response = chat.send_message(query_prompt)
        full_response = response.text
        logger.info(f"Received response from Gemini API: {full_response}")
        
        # Extract just the answer after "A:" and before newline, handling different formats
        for marker in ["A:", "A: "]:
            if marker in full_response:
                answer = full_response.split(marker)[1].split("\n")[0].strip()
                return answer
        return full_response.strip()
        
    except Exception as e:
        error_msg = f"Error calling Gemini API: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def process_wsc_example(
    model: genai.GenerativeModel, 
    example: dict
) -> Dict:
    """Process a single WSC example and get the model's answer."""
    query = construct_query_from_schema(
        example['text'], 
        example['pronoun'], 
        example['options']
    )
    model_response = get_gemini_answer(model, query)
    model_answer = extract_answer(model_response, example['options'])
    correct_answer = example['options'][example['label']]
    
    return {
        'text': example['text'],
        'query': query,
        'model_response': model_response,
        'model_answer': model_answer,
        'correct_answer': correct_answer,
        'is_correct': model_answer == correct_answer
    }

def evaluate_wsc(
    model: genai.GenerativeModel, 
    dataset, 
    num_examples: int = None, 
    delay_between_queries: float = 0
) -> Dict:
    """Evaluate model performance on WSC dataset."""
    results = []
    correct_count = 0
    total_examples = len(dataset) if num_examples is None else min(num_examples, len(dataset))
    
    for i in range(total_examples):
        logger.info(f"Processing example {i+1}/{total_examples}")
        example = dataset[i]
        try:
            result = process_wsc_example(model, example)
            results.append(result)
            if result['is_correct']:
                correct_count += 1
                
            # Add optional delay between queries (in addition to rate limiting)
            if delay_between_queries > 0:
                time.sleep(delay_between_queries)
                
        except Exception as e:
            logger.error(f"Error processing example {i+1}: {str(e)}")
            results.append({
                'text': example['text'],
                'error': str(e),
                'is_correct': False
            })
            
        # Log progress
        accuracy = (correct_count / (i + 1)) * 100
        logger.info(f"Current accuracy: {accuracy:.2f}% ({correct_count}/{i+1})")
    
    # Calculate final statistics
    final_accuracy = (correct_count / total_examples) * 100
    
    return {
        'results': results,
        'total_examples': total_examples,
        'correct_count': correct_count,
        'accuracy': final_accuracy
    }

def print_evaluation_results(eval_results: Dict):
    """Print formatted evaluation results."""
    print("\n=== WSC273 Evaluation Results ===")
    print(f"Total examples processed: {eval_results['total_examples']}")
    print(f"Correct answers: {eval_results['correct_count']}")
    print(f"Accuracy: {eval_results['accuracy']:.2f}%")
    
    print("\nDetailed Results:")
    for i, result in enumerate(eval_results['results'], 1):
        print(f"\nExample {i}:")
        print(f"Text: {result['text']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Model's answer: {result['model_answer']}")
            print(f"Correct answer: {result['correct_answer']}")
            print(f"Correct: {'✓' if result['is_correct'] else '✗'}")

def get_cache_dir() -> Path:
    """Get the cache directory path."""
    cache_dir = Path.home() / '.cache' / 'wsc273'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def load_cached_dataset():
    """Load the dataset from cache if available."""
    cache_file = get_cache_dir() / 'wsc273_dataset.pkl'
    if cache_file.exists():
        try:
            logger.info("Loading dataset from cache...")
            with open(cache_file, 'rb') as f:
                dataset = pickle.load(f)
            logger.info("Successfully loaded dataset from cache")
            return dataset
        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning(f"Failed to load cached dataset: {e}. Will download fresh copy.")
            return None
    return None

def save_dataset_to_cache(dataset):
    """Save the dataset to cache."""
    cache_file = get_cache_dir() / 'wsc273_dataset.pkl'
    try:
        logger.info("Saving dataset to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"Dataset cached successfully at {cache_file}")
    except (OSError, pickle.PicklingError) as e:
        logger.error(f"Failed to cache dataset: {e}")

def main():
    """Main function to run the WSC273 evaluation."""
    try:
        model = setup_api()
        
        # Try to load dataset from cache first
        wsc273_dataset = load_cached_dataset()
        if wsc273_dataset is None:
            # Load dataset from HuggingFace if not in cache
            logger.info("Loading WSC273 dataset from HuggingFace...")
            wsc273_dataset = load_dataset('winograd_wsc', name='wsc273', split='test')
            # Cache the dataset for future use
            save_dataset_to_cache(wsc273_dataset)
        
        # Evaluate on the dataset with rate limiting
        num_examples = None  # Change this number or set to None for full dataset
        
        # Calculate minimum delay needed between queries to stay under rate limit
        min_delay = (60 / 6) - 0.1  # 60 seconds / 6 queries, subtract 0.1 for safety margin
        
        logger.info(f"Starting evaluation with rate limit of 6 queries per minute (minimum {min_delay:.2f}s between queries)")
        eval_results = evaluate_wsc(
            model, 
            wsc273_dataset, 
            num_examples, 
            delay_between_queries=min_delay
        )
        
        # Print results
        print_evaluation_results(eval_results)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()