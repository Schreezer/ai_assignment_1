import logging
from typing import List, Dict
from datasets import load_dataset
import asyncio
import aiohttp
import time
from datetime import datetime
import pickle
from pathlib import Path
import json
import os
from dotenv import load_dotenv
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

def get_output_dir() -> Path:
    """Get the output directory path in current working directory."""
    output_dir = Path.cwd() / "results"
    output_dir.mkdir(exist_ok=True)
    return output_dir

def save_results_to_file(eval_results: Dict) -> Path:
    """Save evaluation results to JSON file."""
    output_dir = get_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"wsc273_results_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    return output_file


API_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
API_KEY = os.getenv('HYPERBOLIC_API_KEY')

HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

class RateLimiter:
    def __init__(self, rate: int, per: float):
        self.rate = rate  # Number of requests
        self.per = per   # Per seconds
        self.allowance = rate  # Current token count
        self.last_check = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            current = time.time()
            time_passed = current - self.last_check
            self.last_check = current
            self.allowance += time_passed * (self.rate / self.per)
            
            if self.allowance > self.rate:
                self.allowance = self.rate
                
            if self.allowance < 1:
                # Wait until we have a token
                wait_time = (1 - self.allowance) * (self.per / self.rate)
                logger.info(f"Rate limit reached, waiting for {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                self.allowance = 0
            else:
                self.allowance -= 1

def query_format_helper(pronoun: str, answers: List[str]) -> str:
    """Format a query about pronoun reference given possible answers."""
    if not pronoun or not all(answers):
        raise ValueError("Pronoun and answers must not be empty")
    return f'In the statement above, does "{pronoun}" refer to {answers[0]} or {answers[1]}?'


def construct_query_from_schema(text: str, pronoun: str, answers: List[str]) -> str:
    """Construct a complete query combining the text and formatted question."""
    if not text or not pronoun:
        raise ValueError("Text and pronoun cannot be empty")
    if len(answers) != 2:
        raise ValueError("Must provide exactly two possible answers")
    return f"{text}\n\n{query_format_helper(pronoun, answers)}"


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


async def get_model_answer_async(query_prompt: str, session: aiohttp.ClientSession, rate_limiter: RateLimiter) -> str:
    """Get answer from Hyperbolic Labs API for a Winograd Schema query asynchronously."""
    try:
        # Wait for rate limiter
        await rate_limiter.acquire()

        messages = [
            {
                "role": "system",
                "content": "You are an expert at resolving pronoun references in text. Answer clearly and directly with the correct referent and also include A: at the end containing the final answer",
            },
            {
                "role": "user",
                "content": "S: The trophy doesn't fit in the brown suitcase because it is too large. Q: Does 'it' refer to 'the trophy' or 'the brown suitcase'?",
            },
            {
                "role": "assistant",
                "content": "After analyzing the sentence, 'it' refers to 'the trophy'. We can determine this because the statement suggests something is too large to fit in the suitcase, which logically means the trophy must be the large object preventing it from fitting. Therefore, A: The trophy.",
            },
            {
                "role": "user",
                "content": "S: The trophy doesn't fit in the brown suitcase because it is too small. Q: Does 'it' refer to 'the trophy' or 'the brown suitcase'?",
            },
            {
                "role": "assistant",
                "content": "'It' refers to 'the brown suitcase'. The statement indicates something is too small, which logically must refer to the container (suitcase) being insufficient to hold the trophy. A: The brown suitcase.",
            },
            {"role": "user", "content": query_prompt},
        ]

        data = {
            "messages": messages,
            "model": "deepseek-ai/DeepSeek-V3",
            "max_tokens": 500,
            "temperature": 0.1,
            "top_p": 0.9,
        }

        logger.info(f"Sending query to API: {query_prompt}")
        async with session.post(API_URL, headers=HEADERS, json=data) as response:
            if response.status == 429:
                logger.warning("Rate limit exceeded, retrying after delay...")
                await asyncio.sleep(10)  # Add extra delay on 429
                return await get_model_answer_async(query_prompt, session, rate_limiter)
                
            response.raise_for_status()
            response_json = await response.json()
            full_response = response_json["choices"][0]["message"]["content"]
            logger.info(f"Received response from API: {full_response}")
            
            for marker in ["A:", "A: "]:
                if marker in full_response:
                    answer = full_response.split(marker)[1].split("\n")[0].strip()
                    return answer

            return full_response.strip()

    except Exception as e:
        error_msg = f"Error calling API: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


async def process_wsc_example_async(example: dict, session: aiohttp.ClientSession, rate_limiter: RateLimiter) -> Dict:
    """Process a single WSC example and get the model's answer asynchronously."""
    query = construct_query_from_schema(
        example["text"], example["pronoun"], example["options"]
    )
    model_response = await get_model_answer_async(query, session, rate_limiter)
    model_answer = extract_answer(model_response, example["options"])
    correct_answer = example["options"][example["label"]]

    return {
        "text": example["text"],
        "query": query,
        "model_response": model_response,
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "is_correct": model_answer == correct_answer,
    }


async def evaluate_wsc_async(
    dataset,
    num_examples: int = None,
    max_concurrent_requests: int = 3,  # Reduced from 6 to 3 for safety
) -> Dict:
    """Evaluate model performance on WSC dataset using async requests."""
    results = []
    correct_count = 0
    total_examples = (
        len(dataset) if num_examples is None else min(num_examples, len(dataset))
    )

    # Initialize rate limiter for 6 requests per minute
    rate_limiter = RateLimiter(rate=6, per=60.0)
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    async def process_with_rate_limit(example, session):
        async with semaphore:
            return await process_wsc_example_async(example, session, rate_limiter)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(total_examples):
            task = process_with_rate_limit(dataset[i], session)
            tasks.append(task)
        
        # Process all examples concurrently
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing example {i+1}: {str(result)}")
                results.append(
                    {"text": dataset[i]["text"], "error": str(result), "is_correct": False}
                )
            else:
                results.append(result)
                if result["is_correct"]:
                    correct_count += 1
            
            accuracy = (correct_count / (i + 1)) * 100
            logger.info(f"Current accuracy: {accuracy:.2f}% ({correct_count}/{i+1})")

    final_accuracy = (correct_count / total_examples) * 100

    return {
        "results": results,
        "total_examples": total_examples,
        "correct_count": correct_count,
        "accuracy": final_accuracy,
    }


def print_evaluation_results(eval_results: Dict):
    """Print formatted evaluation results."""
    print("\n=== WSC273 Evaluation Results ===")
    print(f"Total examples processed: {eval_results['total_examples']}")
    print(f"Correct answers: {eval_results['correct_count']}")
    print(f"Accuracy: {eval_results['accuracy']:.2f}%")

    output_file = save_results_to_file(eval_results)
    print(f"\nResults saved to: {output_file}")

    print("\nDetailed Results:")
    for i, result in enumerate(eval_results["results"], 1):
        print(f"\nExample {i}:")
        print(f"Text: {result['text']}")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Model's answer: {result['model_answer']}")
            print(f"Correct answer: {result['correct_answer']}")
            print(f"Correct: {'✓' if result['is_correct'] else '✗'}")


def get_cache_dir() -> Path:
    """Get the cache directory path."""
    cache_dir = Path.home() / ".cache" / "wsc273"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_cached_dataset():
    """Load the dataset from cache if available."""
    cache_file = get_cache_dir() / "wsc273_dataset.pkl"
    if cache_file.exists():
        try:
            logger.info("Loading dataset from cache...")
            with open(cache_file, "rb") as f:
                dataset = pickle.load(f)
            logger.info("Successfully loaded dataset from cache")
            return dataset
        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning(
                f"Failed to load cached dataset: {e}. Will download fresh copy."
            )
            return None
    return None


def save_dataset_to_cache(dataset):
    """Save the dataset to cache."""
    cache_file = get_cache_dir() / "wsc273_dataset.pkl"
    try:
        logger.info("Saving dataset to cache...")
        with open(cache_file, "wb") as f:
            pickle.dump(dataset, f)
        logger.info(f"Dataset cached successfully at {cache_file}")
    except (OSError, pickle.PicklingError) as e:
        logger.error(f"Failed to cache dataset: {e}")


def main():
    """Main function to run the WSC273 evaluation."""
    try:
        # Try to load dataset from cache first
        wsc273_dataset = load_cached_dataset()
        if wsc273_dataset is None:
            logger.info("Loading WSC273 dataset from HuggingFace...")
            wsc273_dataset = load_dataset("winograd_wsc", name="wsc273", split="test")
            save_dataset_to_cache(wsc273_dataset)

        num_examples = None  # Change this number or set to None for full dataset
        max_concurrent_requests = 3  # Reduced from 6 to 3 for safety

        logger.info(
            f"Starting evaluation with maximum {max_concurrent_requests} concurrent requests"
        )
        eval_results = asyncio.run(evaluate_wsc_async(
            wsc273_dataset, num_examples, max_concurrent_requests
        ))

        print_evaluation_results(eval_results)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
