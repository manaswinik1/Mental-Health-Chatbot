"""Response generation logic for the Mental Health Chatbot."""

from typing import List, Dict
import random
from difflib import SequenceMatcher

from .data_loader import load_data


THRESHOLD = 0.8  # Similarity threshold for matching prompts


def generate_response(user_input: str, data: List[Dict[str, str]]) -> str:
    """Generate an empathetic response to the user input.

    Parameters
    ----------
    user_input : str
        The user's input message.
    data : List[Dict[str, str]]
        Dataset containing prompt-response pairs.

    Returns
    -------
    str
        The generated chatbot response.
    """
    user_input_clean = user_input.strip().lower()

    best_match = None
    highest_ratio = 0.0
    for pair in data:
        ratio = SequenceMatcher(None, user_input_clean, pair["prompt"].lower()).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = pair

    if best_match and highest_ratio >= THRESHOLD:
        return best_match["utterance"]

    return random.choice(data)["utterance"]


def main():
    """Demonstrate simple usage of the response generator."""
    dataset = load_data()
    user_text = "I feel sad today"
    response = generate_response(user_text, dataset)
    print(response)


if __name__ == "__main__":
    main()
