"""
Simple test script to check Hugging Face authentication and dataset loading.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from datasets import load_dataset

# Add project root to Python path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Test Hugging Face authentication and dataset loading."""
    print("Testing Hugging Face authentication and dataset loading...")
    
    # Load environment variables from root .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    print(f"Loading environment from: {env_path}")
    load_dotenv(env_path)
    
    # Get token from environment
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        print("ERROR: HUGGING_FACE_TOKEN not found in environment variables")
        return False
    
    print(f"Found HF token (starts with: {hf_token[:4]}...)")
    
    # Try logging in using huggingface_hub
    try:
        print("\nStep 1: Testing login via huggingface_hub...")
        login(token=hf_token)
        print("✅ Login successful")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False
    
    # Try direct API access
    try:
        print("\nStep 2: Testing API access...")
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        print(f"✅ API access successful - authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ API access failed: {e}")
    
    # Save token to file
    try:
        print("\nStep 3: Saving token to file...")
        token_path = os.path.expanduser("~/.huggingface/token")
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, "w") as f:
            f.write(hf_token)
        print(f"✅ Token saved to {token_path}")
    except Exception as e:
        print(f"❌ Failed to save token: {e}")
    
    # Try loading the dataset
    try:
        print("\nStep 4: Testing dataset loading...")
        dataset = load_dataset(
            "SAA-Lab/writingprompts-pairwise-train",
            use_auth_token=hf_token
        )
        print("✅ Dataset loaded successfully!")
        print(f"Dataset structure: {dataset}")
        
        # Look at first example
        print("\nFirst example keys:", list(dataset["train"][0].keys()))
        
        # Load a small subset
        subset = dataset["train"].select(range(5))
        print(f"\nSuccessfully loaded 5 samples for testing")
        
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed - see errors above") 