from openai import OpenAI
from typing import List, Dict, Any, Optional
import os
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    DEEPINFRA = "deepinfra"
    DEEPSEEK = "deepseek"

class ModelService:
    """
    Unified service for handling OpenAI-compatible model APIs.
    
    This service provides a consistent interface for interacting with multiple LLM providers
    while handling provider-specific requirements and response formats.
    """
    
    # Model mappings for each provider
    MODEL_MAPPINGS = {
        ModelProvider.OPENAI: {
            'gpt-4o-mini': 'gpt-4',  # Using gpt-4 instead of non-existent gpt-4o-mini
            '4o': 'gpt-4',
            'o1': 'gpt-3.5-turbo'  # Using actual model name
        },
        ModelProvider.DEEPINFRA: {
            'qwen-reasoning': 'NovaSky-AI/Sky-T1-32B-Preview',
            'r1': 'deepseek-ai/DeepSeek-R1',
            'v3': 'deepseek-ai/DeepSeek-V3',
            'qwen-base': 'Qwen/Qwen2.5-72B-Instruct'
        },
        ModelProvider.DEEPSEEK: {
            'r1': 'deepseek-reasoner',
            'v3': 'deepseek-chat'
        }
    }

    # API base URLs for each provider
    BASE_URLS = {
        ModelProvider.OPENAI: "https://api.openai.com/v1",
        ModelProvider.DEEPINFRA: "https://api.deepinfra.com/v1/openai",
        ModelProvider.DEEPSEEK: "https://api.deepseek.com/v1"
    }

    # Provider-specific parameters
    PROVIDER_PARAMS = {
        ModelProvider.OPENAI: {
            'max_tokens': 4000,
            'temperature': 0.7
        },
        ModelProvider.DEEPINFRA: {
            'max_tokens': 100000
        },
        ModelProvider.DEEPSEEK: {
            'max_tokens': 4000
        }
    }

    def __init__(self):
        """Initialize the service with empty client cache."""
        self.clients = {}

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        provider: ModelProvider,
        model: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the specified provider and model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            provider: The model provider (OPENAI, DEEPINFRA, or DEEPSEEK)
            model: Model identifier (see MODEL_MAPPINGS for available options)
            api_key: Optional API key (will use environment variable if not provided)
        
        Returns:
            dict: Generated response with 'answer' and optional 'reasoning' keys
            
        Raises:
            ValueError: If required API keys are missing
            Exception: For API errors or other runtime issues
        """
        try:
            # Get or create client and resolve model name
            client = self._get_client(provider, api_key)
            model_name = self._get_model_name(provider, model)

            # Prepare request parameters
            params = {
                'model': model_name,
                'messages': messages,
                'stream': False,
                **self.PROVIDER_PARAMS.get(provider, {})
            }

            # Make API request
            response = client.chat.completions.create(**params)
            content = response.choices[0].message.content

            # Handle special parsing for reasoning-focused models
            if provider in [ModelProvider.DEEPINFRA, ModelProvider.DEEPSEEK] and model in ['r1', 'v3']:
                return self._parse_reasoning_response(content)

            return {'answer': content, 'reasoning': None}

        except Exception as e:
            error_msg = f"Error with {provider.value} API using model {model}: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}

    def _get_client(self, provider: ModelProvider, api_key: Optional[str] = None) -> OpenAI:
        """
        Get or create an OpenAI client for the specified provider.
        
        Args:
            provider: The model provider
            api_key: Optional API key
            
        Returns:
            OpenAI: Configured client for the provider
            
        Raises:
            ValueError: If required API key is not found
        """
        if provider not in self.clients:
            # Get API key from parameters or environment variables
            if not api_key:
                env_var = f"{provider.value.upper()}_API_KEY"
                api_key = os.getenv(env_var)
                if not api_key:
                    raise ValueError(f"{env_var} not found. Please set environment variable or provide api_key parameter")

            self.clients[provider] = OpenAI(
                api_key=api_key,
                base_url=self.BASE_URLS[provider]
            )
        return self.clients[provider]

    def _get_model_name(self, provider: ModelProvider, model: str) -> str:
        """
        Get the actual model name for the provider.
        
        Args:
            provider: The model provider
            model: Short model identifier
            
        Returns:
            str: Full model name for the provider's API
        """
        if model in self.MODEL_MAPPINGS[provider]:
            return self.MODEL_MAPPINGS[provider][model]
        return model

    def _parse_reasoning_response(self, content: str) -> Dict[str, Any]:
        """
        Parse response content from reasoning-focused models.
        
        Args:
            content: Raw response content
            
        Returns:
            dict: Parsed response with answer and optional reasoning
        """
        reasoning = None
        answer = content

        if '<think>' in content and '</think>' in content:
            think_start = content.find('<think>') + len('<think>')
            think_end = content.find('</think>')
            reasoning = content[think_start:think_end].strip()
            answer = content[think_end + len('</think>'):].strip()

        return {
            'answer': answer,
            'reasoning': reasoning
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = ModelService()
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    
    # Test each provider
    for provider in ModelProvider:
        print(f"\nTesting {provider.value}:")
        model = list(ModelService.MODEL_MAPPINGS[provider].keys())[0]
        response = service.generate_response(messages, provider, model)
        print(response) 