import json
import os
import torch
import re
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# Load environment variables
load_dotenv()

# Quantization configuration for efficient inference
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Load DeepSeek-7B model and tokenizer
model_name = "deepseek-ai/deepseek-moe-16b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
)

# Create text generation pipeline
deepseek_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# System prompt remains the same for consistent analysis
SYSTEM_PROMPT = """ 
Strictly follow the format below and analyze text segments from two anonymous debaters (Participant_A and Participant_B). 
Return **ONLY JSON format**, with **NO unnecessary text**.

1. Big Five Inventory (BFI) traits (High/Low for each dimension).
2. Consistency with typical behavior for those traits (Yes/No).
3. Provide a reasoning summary (20 words max).

**Example Output:**
{
    "reasoning": "The speaker is highly agreeable and open but lacks conscientiousness, extraversion, and neuroticism.",
    "predicted_bfi": {
        "Agreeableness": "High",
        "Openness": "High",
        "Conscientiousness": "Low",
        "Extraversion": "Low",
        "Neuroticism": "Low"
    },
    "consistent_with_traits": "Yes"
}

**Rules:**
- **STRICTLY output valid JSON format only**.
- **No disclaimers, markdown, or additional text outside JSON**.
- **Ensure output does not exceed JSON format specifications**.
- **Only reply High/Low and nothing else for classification. It is either High or Low.**

"""

def analyze_bfi_with_deepseek(text, participant):
    """Generates BFI analysis using DeepSeek-7B model."""
    try:
        # Format input with DeepSeek's required template
        prompt = f"<｜begin▁of▁sentence｜>System:\n{SYSTEM_PROMPT}\n\nUser:\nAnalyze {participant}'s text:\n{text}\n\nAssistant:\n"
        
        # Generate response
        response = deepseek_pipeline(
            prompt,
            max_new_tokens=400,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract and clean response
        response_text = response[0]["generated_text"].replace(prompt, "").strip()
        print(f"Raw DeepSeek Output ({participant}):", response_text)
        
        # Extract JSON content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            raise ValueError("JSON not found in response")

    except Exception as e:
        # print(f"Analysis failed for {participant}: {str(e)}")
        print(f"LLaMA Analysis failed for {participant}: {str(e)}")
        return {
            "reasoning": "Analysis failed after retries",
            "predicted_bfi": {
                "Agreeableness": "Error",
                "Openness": "Error",
                "Conscientiousness": "Error",
                "Extraversion": "Error",
                "Neuroticism": "Error"
            },
            "consistent_with_traits": "Error"
        }
# Keep process_discourse() function identical

def process_discourse(input_file, output_file):
    """Processes a JSON file, analyzes personality traits, and saves results incrementally."""
    with open(input_file, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # Open file for incremental saving
    with open(output_file, 'w', encoding="utf-8") as f:
        f.write("[\n")  # Start JSON array

        for i, entry in enumerate(data):
            # Separate messages by persona
            person_one = []
            person_two = []

            for msg in entry['discourse']:
                if "LLaMA-3:" in msg:
                    person_one.append(msg.split(":", 1)[1].strip())
                elif "GPT-4:" in msg:
                    person_two.append(msg.split(":", 1)[1].strip())

            # Analyze each persona using LLaMA-3.1-8B
            entry['bfi_analysis'] = {
                "Person_One": analyze_bfi_with_deepseek("\n".join(person_one), "Participant_A"),
                "Person_Two": analyze_bfi_with_deepseek("\n".join(person_two), "Participant_B")
            }

            # Save each entry incrementally
            f.write(json.dumps(entry, indent=2, ensure_ascii=False))
            if i < len(data) - 1:
                f.write(",\n")  # Add comma for JSON array format

        f.write("\n]")  # Close JSON array properly

if __name__ == "__main__":
    process_discourse("/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/final_discourse_ptvsllama_combinations.json", "bfi_analysis_deepseek_gptvsllama.json")
