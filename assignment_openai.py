import logging
from typing import List, Dict
from datasets import load_dataset
import time
import pickle
from pathlib import Path
from ratelimit import limits, sleep_and_retry
from openai import OpenAI
import os
from dotenv import load_dotenv
import asyncio
from asyncio import Semaphore
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=api_key)

def query_format_helper(pronoun: str, answers: List[str]) -> str:
    if not pronoun or not all(answers):
        raise ValueError("Pronoun and answers must not be empty")
    return f'In the statement above, does "{pronoun}" refer to {answers[0]} or {answers[1]}?'

def construct_query_from_schema(text: str, pronoun: str, answers: List[str]) -> str:
    if not text or not pronoun:
        raise ValueError("Text and pronoun cannot be empty")
    if len(answers) != 2:
        raise ValueError("Must provide exactly two possible answers")
    return f"{text}\n\n{query_format_helper(pronoun, answers)}"

def extract_answer(response: str, options: List[str]) -> str:
    response_lower = response.lower()
    options_lower = [opt.lower() for opt in options]

    for i, option in enumerate(options_lower):
        if option in response_lower:
            return options[i]

    for i, option in enumerate(options_lower):
        words = option.split()
        if any(word in response_lower for word in words):
            return options[i]

    logger.warning(f"Could not match response '{response}' to any option in {options}")
    return options[0]

async def get_model_answer_async(query_prompt: str, semaphore: Semaphore) -> str:
    async with semaphore:
        try:
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

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=500,
                )
            )

            full_response = response.choices[0].message.content
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

async def process_wsc_example_async(example: dict, semaphore: Semaphore) -> Dict:
    query = construct_query_from_schema(
        example["text"], example["pronoun"], example["options"]
    )
    model_response = await get_model_answer_async(query, semaphore)
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

async def evaluate_wsc_async(dataset, num_examples: int = None) -> Dict:
    results = []
    correct_count = 0
    total_examples = len(dataset) if num_examples is None else min(num_examples, len(dataset))
    
    # Create a semaphore to limit concurrent API calls (adjust the value based on your rate limits)
    semaphore = Semaphore(10)  # Allow 10 concurrent requests
    
    async def process_example(i: int):
        nonlocal correct_count
        logger.info(f"Processing example {i+1}/{total_examples}")
        example = dataset[i]
        try:
            result = await process_wsc_example_async(example, semaphore)
            if result["is_correct"]:
                correct_count += 1
            return result
        except Exception as e:
            logger.error(f"Error processing example {i+1}: {str(e)}")
            return {"text": example["text"], "error": str(e), "is_correct": False}

    # Create tasks for all examples
    tasks = [process_example(i) for i in range(total_examples)]
    results = await asyncio.gather(*tasks)

    final_accuracy = (correct_count / total_examples) * 100

    return {
        "results": results,
        "total_examples": total_examples,
        "correct_count": correct_count,
        "accuracy": final_accuracy,
    }

def print_evaluation_results(eval_results: Dict):
    print("\n=== WSC273 Evaluation Results ===")
    print(f"Total examples processed: {eval_results['total_examples']}")
    print(f"Correct answers: {eval_results['correct_count']}")
    print(f"Accuracy: {eval_results['accuracy']:.2f}%")

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
    cache_dir = Path.home() / ".cache" / "wsc273"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def save_evaluation_results(eval_results: Dict):
    results_dir = get_cache_dir() / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"wsc273_results_{timestamp}.json"
    
    try:
        logger.info(f"Saving evaluation results to {results_file}")
        with open(results_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def load_cached_dataset():
    cache_file = get_cache_dir() / "wsc273_dataset.pkl"
    if cache_file.exists():
        try:
            logger.info("Loading dataset from cache...")
            with open(cache_file, "rb") as f:
                dataset = pickle.load(f)
            logger.info("Successfully loaded dataset from cache")
            return dataset
        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning(f"Failed to load cached dataset: {e}. Will download fresh copy.")
            return None
    return None

def save_dataset_to_cache(dataset):
    cache_file = get_cache_dir() / "wsc273_dataset.pkl"
    try:
        logger.info("Saving dataset to cache...")
        with open(cache_file, "wb") as f:
            pickle.dump(dataset, f)
        logger.info(f"Dataset cached successfully at {cache_file}")
    except (OSError, pickle.PicklingError) as e:
        logger.error(f"Failed to cache dataset: {e}")

def main():
    try:
        wsc273_dataset = load_cached_dataset()
        if wsc273_dataset is None:
            logger.info("Loading WSC273 dataset from HuggingFace...")
            wsc273_dataset = load_dataset("winograd_wsc", name="wsc273", split="test")
            save_dataset_to_cache(wsc273_dataset)

        num_examples = None  # Change this number or set to None for full dataset

        logger.info("Starting evaluation...")
        eval_results = asyncio.run(evaluate_wsc_async(wsc273_dataset, num_examples))

        print_evaluation_results(eval_results)
        save_evaluation_results(eval_results)  # Save results to file

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()