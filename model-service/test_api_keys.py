import os
from model_service import ModelService, ModelProvider

def mask_api_key(key: str, show_chars: int = 4) -> str:
    """Safely mask an API key showing only first and last few characters"""
    if not key:
        return "Not found"
    if len(key) <= show_chars * 2:
        return "***"
    return f"{key[:show_chars]}...{key[-show_chars:]}"

def test_api_keys():
    """Test if API keys are properly loaded from environment variables"""
    print("\nTesting API key availability:")
    
    # Check environment variables
    openai_key = os.getenv('OPENAI_API_KEY')
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    deepinfra_key = os.getenv('DEEPINFRA_TOKEN')
    
    print(f"OpenAI API Key: {mask_api_key(openai_key)}")
    print(f"DeepSeek API Key: {mask_api_key(deepseek_key)}")
    print(f"DeepInfra Token: {mask_api_key(deepinfra_key)}")
    
    # Test model service initialization
    service = ModelService()
    
    # Test each provider
    test_message = [{"role": "user", "content": "Test message"}]
    
    print("\nTesting provider initialization:")
    
    try:
        client = service._get_client(ModelProvider.OPENAI)
        print(f"✓ OpenAI client initialized successfully with key: {mask_api_key(client.api_key)}")
    except Exception as e:
        print(f"✗ OpenAI client failed: {str(e)}")
        
    try:
        client = service._get_client(ModelProvider.DEEPSEEK)
        print(f"✓ DeepSeek client initialized successfully with key: {mask_api_key(client.api_key)}")
    except Exception as e:
        print(f"✗ DeepSeek client failed: {str(e)}")
        
    try:
        client = service._get_client(ModelProvider.DEEPINFRA)
        print(f"✓ DeepInfra client initialized successfully with key: {mask_api_key(client.api_key)}")
    except Exception as e:
        print(f"✗ DeepInfra client failed: {str(e)}")

if __name__ == "__main__":
    test_api_keys() 