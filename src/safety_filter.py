"""Safety filter for chatbot responses."""

from typing import List

BANNED_WORDS: List[str] = ["suicide", "kill", "die", "hate", "worthless"]
DEFAULT_MESSAGE = (
    "I'm here to support you. Please consider reaching out to someone you trust."
)


def filter_response(response: str) -> str:
    """Check the response for banned words and return a safe message.

    Parameters
    ----------
    response : str
        The chatbot-generated response.

    Returns
    -------
    str
        The filtered response.
    """
    lower_resp = response.lower()
    for word in BANNED_WORDS:
        if word in lower_resp:
            return DEFAULT_MESSAGE
    return response


if __name__ == "__main__":
    sample = "I feel like I want to die"
    print(filter_response(sample))
