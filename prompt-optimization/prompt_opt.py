import os
import sys
import textgrad as tg
from datasets import load_dataset
import random
import re
import datetime
import json

# Add project root to Python path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from root .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Try to authenticate with Hugging Face
try:
    from huggingface_hub import login
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if hf_token:
        print(f"Found HF token (starts with: {hf_token[:4]}...)")
        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub")
except Exception as e:
    print(f"Non-critical error with HF authentication: {e}")

def load_samples(num_samples=10):
    """Load samples from the WritingPrompts dataset"""
    try:
        # Try loading the dataset
        dataset = load_dataset("SAA-Lab/writingprompts-pairwise-train")
        print("Dataset loaded successfully!")
        
        # Take a few samples for testing
        validation_set = dataset["train"].select(range(num_samples))
        print(f"Successfully loaded {num_samples} samples for testing")
        return validation_set
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def format_example(example):
    """Format an example with A/B stories"""
    formatted = {
        "prompt": example["prompt"],
        "story_a": example["chosen"],
        "story_b": example["rejected"],
        "correct_answer": "a"  # The chosen story is in position A
    }
    
    # Randomly shuffle to avoid position bias (50% chance)
    if random.random() < 0.5:
        formatted["story_a"], formatted["story_b"] = formatted["story_b"], formatted["story_a"]
        formatted["correct_answer"] = "b"  # Now the chosen story is in position B
        
    return formatted

def extract_preference(text):
    """Extract preference (A or B) from text with multiple patterns
    
    Returns:
        tuple: (preference, method) where preference is 'a', 'b', or 'unknown'
               and method is a string indicating which extraction method succeeded
    """
    # Try explicit "Preferred: X" format first (still most reliable)
    match = re.search(r"Preferred:\s*([AB])", text, re.IGNORECASE)
    if match:
        return match.group(1).lower(), "explicit_preferred_tag"
    
    # NEW: Handle the "**A/B Strengths**" format we're seeing in responses
    if "**A Strengths**" in text and "**B Strengths**" in text:
        # Count number of strength bullet points for each
        a_bullets = len(re.findall(r"\*\*A Strengths\*\*:?\s*(?:\n\s*-[^\n]+)+", text, re.IGNORECASE))
        b_bullets = len(re.findall(r"\*\*B Strengths\*\*:?\s*(?:\n\s*-[^\n]+)+", text, re.IGNORECASE))
        
        # Look for explicit preference statements in the conclusion
        if "conclusion" in text.lower() or "overall" in text.lower():
            conclusion_part = text.split("**", 1)[0] if text.startswith("**") else ""
            conclusion_part += re.split(r"\*\*(?:Conclusion|Overall|Final|Summary)\*\*", text, flags=re.IGNORECASE)[-1]
            
            # Check for preference in conclusion
            for pattern in [
                r"(?:response|option)\s+([AB])\s+(?:is|shows|displays|exhibits|demonstrates)",
                r"(?:I\s+)?(?:choose|select|prefer|recommend)\s+(?:response\s+)?([AB])",
                r"(?:response|option)\s+([AB])\s+(?:is|seems)\s+(?:more|better|stronger)"
            ]:
                match = re.search(pattern, conclusion_part, re.IGNORECASE)
                if match:
                    return match.group(1).lower(), "strengths_format_conclusion"
        
        # Analyze the sentiment in each section
        a_section = re.findall(r"\*\*A Strengths\*\*:?.*?(?=\*\*B Strengths\*\*|\Z)", text, re.DOTALL | re.IGNORECASE)
        b_section = re.findall(r"\*\*B Strengths\*\*:?.*?(?=\*\*|\Z)", text, re.DOTALL | re.IGNORECASE)
        
        a_positive = sum(1 for _ in re.finditer(r"(?:strong|rich|vivid|detailed|excellent|effective|compelling|engaging)", 
                                              "".join(a_section), re.IGNORECASE))
        b_positive = sum(1 for _ in re.finditer(r"(?:strong|rich|vivid|detailed|excellent|effective|compelling|engaging)", 
                                              "".join(b_section), re.IGNORECASE))
        
        # Compare the number of positive attributes
        if a_positive > b_positive + 1:  # Require a clearer difference
            return "a", "strengths_format_sentiment"
        elif b_positive > a_positive + 1:
            return "b", "strengths_format_sentiment"
        
        # Compare the number of bullet points
        if a_bullets > b_bullets + 2:  # Require a clearer difference in bullets
            return "a", "strengths_format_bullets"
        elif b_bullets > a_bullets + 2:
            return "b", "strengths_format_bullets"
    
    # Look for preference in "Analysis" or bulleted format that the model seems to be using
    if "**Analysis:**" in text or "Response A:" in text:
        # Look for conclusion statements after comparison sections
        conclusion_indicators = [
            r"overall[^.]*(?:response|option)\s+([AB])\s+(?:is|shows|displays|exhibits|demonstrates)",
            r"(?:I\s+)?(?:conclude|determine|find|believe)[^.]*(?:response|option)\s+([AB])",
            r"(?:In\s+)?(?:conclusion|summary)[^.]*(?:response|option)\s+([AB])",
            r"(?:response|option)\s+([AB])\s+(?:is|shows|displays|demonstrates)[^.]*(?:more creative|stronger|better)",
            r"(?:I|we)\s+(?:prefer|choose|select|pick)\s+(?:response|option)?\s*([AB])"
        ]
        
        for i, pattern in enumerate(conclusion_indicators):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower(), f"analysis_conclusion_{i+1}"
        
        # If no conclusion statement, try to determine from the comparative language
        a_positive = len(re.findall(r"Response A[^.]*(?:more|better|stronger|effective|creative|rich|vivid)", text, re.IGNORECASE))
        b_positive = len(re.findall(r"Response B[^.]*(?:more|better|stronger|effective|creative|rich|vivid)", text, re.IGNORECASE))
        a_negative = len(re.findall(r"Response A[^.]*(?:less|weaker|lacks|fails|limited|basic)", text, re.IGNORECASE))
        b_negative = len(re.findall(r"Response B[^.]*(?:less|weaker|lacks|fails|limited|basic)", text, re.IGNORECASE))
        
        # Calculate net sentiment
        a_score = a_positive - a_negative
        b_score = b_positive - b_negative
        
        if a_score > b_score and a_score > 0:
            return "a", "analysis_sentiment_comparison"
        elif b_score > a_score and b_score > 0:
            return "b", "analysis_sentiment_comparison"
    
    # Look for preference between "Reasoning:" and "Confidence:" sections
    if "Reasoning:" in text:
        # Get the text between "Reasoning:" and "Confidence:" (or end of string)
        reasoning_text = text.split("Reasoning:", 1)[1]
        if "Confidence:" in reasoning_text:
            reasoning_text = reasoning_text.split("Confidence:", 1)[0]
            
        # Look for preference indicators in the reasoning section
        match = re.search(r"(?:response|option)\s+([AB])\s+(?:is|shows|displays|exhibits|demonstrates)", reasoning_text, re.IGNORECASE)
        if match:
            return match.group(1).lower(), "reasoning_section_indicators"
            
        # Look for a preference statement after reasoning
        match = re.search(r"(?:I\s+(?:choose|select|prefer)|Therefore,\s+(?:I\s+)?(?:choose|select|prefer))\s+(?:response\s+)?([AB])", reasoning_text, re.IGNORECASE)
        if match:
            return match.group(1).lower(), "reasoning_section_conclusion"
    
    # Try "I prefer X" or "I choose X" format
    match = re.search(r"I\s+(?:prefer|choose|select|pick)\s+(?:response\s+)?([AB])", text, re.IGNORECASE)
    if match:
        return match.group(1).lower(), "explicit_preference_statement"
        
    # Try "X is better/more creative"
    match = re.search(r"(?:Response\s+)?([AB])\s+is\s+(?:better|more creative|preferred)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower(), "comparative_statement"
        
    # Check for any sentences that mention A or B with positive markers
    a_sentiment = len(re.findall(r"(?:response\s+)?A.{0,30}(?:better|creative|engaging|prefer)", text, re.IGNORECASE))
    b_sentiment = len(re.findall(r"(?:response\s+)?B.{0,30}(?:better|creative|engaging|prefer)", text, re.IGNORECASE))
    
    if a_sentiment > b_sentiment and a_sentiment > 0:
        return "a", "proximity_sentiment"
    elif b_sentiment > a_sentiment and b_sentiment > 0:
        return "b", "proximity_sentiment"
        
    # Last resort - check which letter appears more frequently in context
    a_count = len(re.findall(r"(?:^|\W)A(?:\W|$)", text))
    b_count = len(re.findall(r"(?:^|\W)B(?:\W|$)", text))
    
    if a_count > b_count * 2:  # Require significantly more mentions to avoid noise
        return "a", "frequency_analysis"
    elif b_count > a_count * 2:
        return "b", "frequency_analysis"
        
    return "unknown", "no_method_succeeded"

def improve_prompt(engine, current_prompt, is_correct, prediction_text):
    """Improve the system prompt based on the model's performance"""
    
    # Instruction for the model to improve the prompt
    instruction = f"""You are improving a system prompt for a creative writing judge.

CURRENT PROMPT:
{current_prompt}

RECENT PERFORMANCE:
The model's prediction was {"CORRECT" if is_correct else "INCORRECT"}.
Here's the model's output: 
{prediction_text[:500]}...

TASK:
You have complete freedom to rewrite the prompt to make it better at:
1. Evaluating creative dimensions (you can modify, add, or remove dimensions as you see fit)
2. Guiding the model to choose the more creative response
3. Structuring the output format for clear reasoning
4. Being clear, concise, and effective

IMPORTANT CONSTRAINTS:
1. The prompt MUST be under 1000 characters total - brevity is essential
2. You MUST include VERY EXPLICIT format instructions for the output
3. The output format MUST include "Preferred: [A or B]" - this is CRITICAL
4. The prompt MUST emphasize following the exact output format

BE CREATIVE - don't be afraid to substantially change the prompt if you think it will improve performance.
You can reorganize sections, change wording, add examples, or modify instructions.

DO NOT include explanations, just provide the revised prompt text.
"""
    
    try:
        # Generate improved prompt with much higher token limit
        response = engine.generate(instruction, max_tokens=2000, temperature=0.8)
        
        # Less restrictive cleaning
        lines = response.strip().split('\n')
        
        # Skip any obvious explanatory lines but keep most content
        clean_lines = []
        in_explanation = False
        for line in lines:
            # Skip obvious headers or explanations
            if line.strip().startswith('```') or line.strip().startswith('Here is'):
                in_explanation = not in_explanation
                continue
                
            if in_explanation:
                continue
                
            # Skip lines that are clearly just commentary
            if any(x in line.lower() for x in ['here is the improved prompt', 'revised prompt', 'new prompt:']):
                continue
                
            clean_lines.append(line)
            
        new_prompt = '\n'.join(clean_lines).strip()
        
        # Much less restrictive validation - ensure it's not too short or too long
        if 100 <= len(new_prompt) <= 1000:  # Stricter upper limit of 1000 chars
            # IMPORTANT: Verify the prompt contains format instructions
            if "Preferred:" in new_prompt or "preferred:" in new_prompt:
                # Only consider it a new prompt if it's meaningfully different (>10% change)
                if len(set(new_prompt) - set(current_prompt)) / len(set(current_prompt)) > 0.10:
                    print(f"  New prompt length: {len(new_prompt)} characters")
                    return new_prompt
                else:
                    print("  Generated prompt was too similar to current prompt")
            else:
                print("  Generated prompt missing critical 'Preferred:' format directive")
        else:
            print(f"  Generated prompt length ({len(new_prompt)}) outside acceptable range (100-1000)")
            
    except Exception as e:
        print(f"Error improving prompt: {e}")
    
    # Return original prompt if there was an error or the new one didn't meet criteria
    return current_prompt

def main():
    # Load more samples for better optimization
    dataset = load_samples(30)  # Reduced from 50 to 30 for faster testing
    
    # Initialize the model
    engine_name = "gpt-4o-mini"
    engine = tg.get_engine(engine_name)
    
    # Our initial system prompt - made more explicit about format
    initial_prompt = """You're evaluating creative writing responses A and B.

Compare them based on these dimensions:
- Imagery: vivid descriptions and sensory details
- Tension: dramatic interest and conflict
- Pattern: structural elements and composition
- Energy: engaging style and dynamic writing
- Insight: meaningful ideas and depth

IMPORTANT: Your answer MUST use EXACTLY this format:
Reasoning: [brief comparison]
Preferred: [A or B] (state which one is better)
Confidence: [0-1 score]

Example format:
Reasoning: Response B has stronger imagery and tension.
Preferred: B
Confidence: 0.8"""
    
    current_prompt = initial_prompt
    
    # Track performance
    total_examples = 0
    correct_examples = 0
    all_results = []
    extraction_methods = {}  # Track which methods were used for extraction
    improvements_made = 0 # Track number of improvements made
    format_violations = 0  # Track format compliance issues
    
    # Process examples
    print("\n--- Testing creative writing evaluation ---\n")
    
    for i, example in enumerate(dataset):
        # Format example
        formatted = format_example(example)
        
        # Format input for the model - limit to 300 chars to make it more challenging
        user_input = f"""WRITING PROMPT:
{formatted['prompt']}

RESPONSE A:
{formatted['story_a'][:300]}...

RESPONSE B:
{formatted['story_b'][:300]}..."""

        # Get prediction
        print(f"Example {i+1}/{len(dataset)}: Processing...")
        try:
            # Call model with current prompt
            response = engine.generate(current_prompt + "\n\n" + user_input, max_tokens=800)
            prediction_text = response.strip()
            
            # Extract preference
            preferred, method = extract_preference(prediction_text)
            
            # Track extraction method usage
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
            
            # Track format violations
            if method != "explicit_preferred_tag" and "Preferred:" not in prediction_text:
                format_violations += 1
            
            # Evaluate correctness
            is_correct = preferred == formatted['correct_answer']
            if is_correct:
                correct_examples += 1
            total_examples += 1
            
            # Print results with colorized output
            print(f"  Model chose: {preferred.upper()}")
            print(f"  Correct answer: {formatted['correct_answer'].upper()}")
            print(f"  Match: {'✓' if is_correct else '✗'}")
            print(f"  Extraction method: {method}")
            print(f"  Format compliance: {'✓' if 'Preferred:' in prediction_text else '✗'}")
            print(f"  Response snippet: {prediction_text[:150]}...")
            
            # Try to improve prompt on errors or periodically
            if not is_correct or method == "no_method_succeeded" or i % 5 == 0:
                print("  Attempting to improve prompt...")
                new_prompt = improve_prompt(engine, current_prompt, is_correct, prediction_text)
                
                if new_prompt != current_prompt:
                    improvements_made += 1
                    print(f"  Prompt updated (improvement #{improvements_made})")
                    # Print a snippet of what changed
                    print(f"  New prompt begins with: {new_prompt[:100]}...")
                    current_prompt = new_prompt
                else:
                    print("  No changes made to prompt")
            
            # Print current prompt (just the beginning)
            print(f"\nCurrent prompt (start):")
            print(f"{current_prompt[:100]}...")
            print("-" * 60)
            
            # Save example result
            all_results.append({
                "example_id": i,
                "preferred": preferred,
                "correct_answer": formatted['correct_answer'],
                "is_correct": is_correct,
                "extraction_method": method,
                "follows_format": "Preferred:" in prediction_text,
                "prediction_text": prediction_text[:300]
            })
            
            # Calculate and print running accuracy
            running_accuracy = correct_examples / total_examples
            print(f"Running accuracy: {running_accuracy:.2%} ({correct_examples}/{total_examples})")
            
            # Update on format compliance
            format_compliance = total_examples - format_violations
            print(f"Format compliance: {format_compliance/total_examples:.2%} ({format_compliance}/{total_examples})")
            
        except Exception as e:
            print(f"  Error processing example: {e}")
            total_examples += 1
    
    # Calculate accuracy
    accuracy = correct_examples / max(1, total_examples)
    format_compliance_rate = (total_examples - format_violations) / max(1, total_examples)
    
    # Save the optimized prompt
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"optimized_prompt_{timestamp}.txt"
    )
    
    with open(output_path, "w") as f:
        f.write("# Creative Writing Evaluation Prompts\n\n")
        
        # Save the original prompt first
        f.write("## Original Prompt\n\n")
        f.write(initial_prompt)
        f.write("\n\n")
        
        # Then save the optimized prompt
        f.write("## Optimized Prompt\n\n")
        f.write(current_prompt)
        f.write("\n\n")
        
        # Add usage instructions
        f.write("## Usage Instructions\n")
        f.write("When using this prompt, append the writing prompt and responses in this format:\n\n")
        f.write("```\n")
        f.write("WRITING PROMPT:\n")
        f.write("[your writing prompt here]\n\n")
        f.write("RESPONSE A:\n")
        f.write("[first creative response here]\n\n")
        f.write("RESPONSE B:\n")
        f.write("[second creative response here]\n")
        f.write("```\n\n")
        
        f.write("The model should respond with exactly this format:\n\n")
        f.write("```\n")
        f.write("Reasoning: [comparative analysis]\n")
        f.write("Preferred: [A or B]\n")
        f.write("Confidence: [0-1 score]\n")
        f.write("```\n\n")
        
        f.write(f"Model used: {engine_name}\n")
        f.write(f"Accuracy: {correct_examples}/{total_examples} ({accuracy*100:.1f}%)\n")
        f.write(f"Format compliance: {total_examples - format_violations}/{total_examples} ({format_compliance_rate*100:.1f}%)\n")
        f.write(f"Date created: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")
        
        # Also save the training results
        f.write("\n\n## Training Results\n")
        for i, result in enumerate(all_results):
            f.write(f"Example {i+1}: Model chose {result['preferred'].upper()}, ")
            f.write(f"Correct answer was {result['correct_answer'].upper()}, ")
            f.write(f"Result: {'✓' if result['is_correct'] else '✗'}, ")
            f.write(f"Method: {result['extraction_method']}, ")
            f.write(f"Format: {'✓' if result.get('follows_format', False) else '✗'}\n")
        
        # Add extraction method statistics
        f.write("\n\n## Extraction Method Statistics\n")
        for method, count in sorted(extraction_methods.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{method}: {count} times ({count/total_examples*100:.1f}%)\n")
        
        # Add format compliance summary
        f.write(f"\nFormat compliance rate: {format_compliance_rate*100:.1f}%\n")
    
    print(f"\nOptimized prompt saved to: {output_path}")
    print(f"Accuracy: {accuracy:.2%} ({correct_examples}/{total_examples})")
    print(f"Format compliance: {format_compliance_rate:.2%} ({total_examples - format_violations}/{total_examples})")
    
    # Print extraction method statistics
    print("\nExtraction Method Statistics:")
    for method, count in sorted(extraction_methods.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method}: {count} times ({count/total_examples*100:.1f}%)")

if __name__ == "__main__":
    main()

