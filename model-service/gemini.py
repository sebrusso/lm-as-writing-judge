import google.generativeai as genai
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_gemini_response(messages: List[Dict[str, str]], api_key: str = None, model: str = 'thinking') -> Dict[str, Any]:
    """
    Generate a response using the Gemini API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        api_key: Gemini API key (optional - will use GEMINI_API_KEY env var if not provided)
        model: Model to use - 'thinking' or 'non-thinking'
    
    Returns:
        dict: Generated response with 'answer' and 'reasoning' keys
    """
    # Configure API key
    if not api_key:
        if 'GEMINI_API_KEY' not in os.environ:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable or provide api_key parameter")
        api_key = os.environ['GEMINI_API_KEY']

    genai.configure(api_key=api_key)

    # Initialize model
    model_name = "gemini-2.0-flash-thinking-exp-01-21" if model == 'thinking' else "gemini-2.0-flash-exp"
    model = genai.GenerativeModel(model_name)

    # Convert messages to chat history format
    history = []
    for message in messages[:-1]: # Exclude last message
        history.append({
            "role": "user" if message["role"] == "user" else "model",
            "parts": [message["content"]]
        })

    # Start chat with history and get response
    chat = model.start_chat(history=history)
    response = chat.send_message(messages[-1]["content"])

    return {
        'answer': response.text,
        'reasoning': None  # Gemini doesn't provide explicit reasoning
    }


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    print(generate_gemini_response(messages, model="thinking"))