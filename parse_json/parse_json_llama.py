import json
import os
import torch
import re
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# Load environment variables
load_dotenv()

# Quantization settings for LLaMA-3.1-8B
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Load LLaMA-3.1-8B model and tokenizer with quantization
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
llama_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.3-70B-Instruct",
    device_map="auto",
    quantization_config=quantization_config
)

# Create a text-generation pipeline with LLaMA-3.1-8B
llama_pipeline = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    device_map="auto"
)

# **Updated System Prompt for Analysis**
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

# **Error Handling:** If analysis fails, return:
# {
#     "reasoning": "Analysis failed after retries",
#     "predicted_bfi": {
#         "Agreeableness": "Error",
#         "Openness": "Error",
#         "Conscientiousness": "Error",
#         "Extraversion": "Error",
#         "Neuroticism": "Error"
#     },
#     "consistent_with_traits": "Error"
# }

def analyze_bfi_with_llama(text, participant):
    """Generates BFI analysis using LLaMA-3.1-8B model with chat template."""
    try:
        # Apply chat template to format the conversation properly
        prompt = llama_tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": f"Analyze {participant}'s text:\n{text}\n\nResponse:"}],
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response with LLaMA-3.1-8B
        response = llama_pipeline(
            prompt,
            max_new_tokens=400,
            temperature=0.1,  # Lower randomness for structured JSON
            top_p=0.9,
            do_sample=True,
            eos_token_id=llama_tokenizer.eos_token_id,  # Ensures stopping at end-of-sequence token
            pad_token_id=llama_tokenizer.eos_token_id  # Avoids unwanted padding tokens
        )

        # Extract response text
        response_text = response[0]["generated_text"].replace(prompt, "").strip()
        print(f"Raw LLaMA Output ({participant}):", response_text)

        # Remove markdown formatting (if present)
        response_text = response_text.replace("```json", "").replace("```", "").strip()

        # Extract JSON portion only
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_response = json_match.group(0)
            return json.loads(json_response)
        else:
            raise ValueError("Invalid JSON format in LLaMA response")

    except Exception as e:
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
                if "DeepSeek:" in msg:
                    person_one.append(msg.split(":", 1)[1].strip())
                elif "GPT-4:" in msg:
                    person_two.append(msg.split(":", 1)[1].strip())

            # Analyze each persona using LLaMA-3.1-8B
            entry['bfi_analysis'] = {
                "Person_One": analyze_bfi_with_llama("\n".join(person_one), "Participant_A"),
                "Person_Two": analyze_bfi_with_llama("\n".join(person_two), "Participant_B")
            }

            # Save each entry incrementally
            f.write(json.dumps(entry, indent=2, ensure_ascii=False))
            if i < len(data) - 1:
                f.write(",\n")  # Add comma for JSON array format

        f.write("\n]")  # Close JSON array properly

if __name__ == "__main__":
    process_discourse("/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/final_discorse_gptvsdeepseek.json", "bfi_analysis_llama_gptvsdeepseek.json")
