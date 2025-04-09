import openai
import transformers
import torch
import json
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found in environment")

# DeepSeek configuration
def initialize_deepseek():
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-llm-7b-chat",
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        ),
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

deepseek_pipeline = initialize_deepseek()

# Debate configuration
TRAITS = [{
    "Agreeableness": "High", 
    "Openness": "Low",
    "Conscientiousness": "High",
    "Extraversion": "Low",
    "Neuroticism": "High"
}]

TOPICS = ["Should artificial intelligence be used in military applications?"]

def build_system_prompt(topic: str, traits: dict) -> str:
    return f"""<|im_start|>system
You are a debate participant. Respond DIRECTLY with your argument.

Topic: {topic}
Personality: {json.dumps(traits)}

Rules:
- NO thinking process
- NO justification
- ONLY pure argument
- MAX 2 sentences
- Use {traits['Agreeableness']} agreeableness

Bad Example: "I think we should consider... [explanation]"
Good Example: "Autonomous systems risk escalation cycles."<|im_end|>
"""
def generate_gpt_response(session: list) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=session,
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"GPT Error: {str(e)}")
        return "[GPT Response Failed]"

def generate_deepseek_response(prompt: str) -> str:
    try:
        outputs = deepseek_pipeline(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=deepseek_pipeline.tokenizer.eos_token_id,
        )
        
        if not outputs:
            return "[Empty Response]"
            
        return outputs[0]['generated_text'].split("<|im_end|>")[-1].strip()
        
    except Exception as e:
        print(f"Generation Error: {str(e)}")
        return "[Response Failed]"

def validate_response(text: str) -> str:
    text = str(text).split("<|im_end|>")[0].strip()
    text = text.replace("Assistant:", "").replace("User:", "").strip()
    
    if text and text[-1] not in {".", "!", "?"}:
        text += "."
        
    return text[:500]

def conduct_debate(topic: str, traits: dict, rounds: int = 3) -> list:
    debate_log = []
    system_prompt = build_system_prompt(topic, traits)
    
    # Initialize message histories
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Present your opening argument"}
    ]

    # Generate opening statement
    try:
        ds_response = generate_deepseek_response("\n".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]))
        clean_response = validate_response(ds_response)
        debate_log.append(f"DeepSeek: {clean_response}")
        messages.append({"role": "assistant", "content": clean_response})
    except Exception as e:
        print(f"Opening Error: {str(e)}")
        return debate_log

    # Debate rounds
    for _ in range(rounds):
        # GPT response
        gpt_response = generate_gpt_response(messages)
        debate_log.append(f"GPT: {gpt_response}")
        messages.append({"role": "user", "content": gpt_response})

        # DeepSeek response
        ds_response = generate_deepseek_response("\n".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]))
        clean_response = validate_response(ds_response)
        debate_log.append(f"DeepSeek: {clean_response}")
        messages.append({"role": "assistant", "content": clean_response})

    return debate_log

def save_debates(output_file: str = "debate_logs.json"):
    debates = []
    for traits in TRAITS:
        for topic in TOPICS:
            print(f"Debating: {topic}")
            debates.append({
                "topic": topic,
                "traits": traits,
                "discourse": conduct_debate(topic, traits)
            })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(debates, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    save_debates()