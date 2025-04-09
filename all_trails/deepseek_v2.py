import openai
import transformers
import torch
import json
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pdb

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found in environment variables")

# Initialize DeepSeek-67B with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-67b-chat")
deepseek_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-llm-67b-chat",
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

deepseek_pipeline = transformers.pipeline(
    "text-generation",
    model=deepseek_model,
    tokenizer=deepseek_tokenizer,
    device_map="auto"
)


# pdb.set_trace()

 #Load the JSON data
with open("debate_topics_traits.json", "r") as file:
    data = json.load(file)

TOPICS = data["debate_topics"]
TRAITS = data["personality_traits"]

def build_system_prompt(topic: str, traits: dict) -> str:
    return (
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"You are participating in a structured debate on: '{topic}\n"
        f"your Personality traits are: {json.dumps(traits)}\n"
        "Rules:\n"
        "- Maintain this personality traits (DONOT EXPLIITLY CMENTION IN TEXT) at all time during your conversation\n"
        "- Keep responses under 50 words\n"
        "- Maintain logical coherence\n"
        "- Address previous points directly\n"
        "- Use natural conversational English<|eot_id|>"
    )


def generate_gpt_response(session: list) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=session,
            temperature=0.72,
            max_tokens=150,
            frequency_penalty=0.5
        )
        content = response.choices[0].message.content.strip()
        return validate_response(content)
    except Exception as e:
        print(f"GPT Error: {str(e)}")
        return "[GPT Response Failed]"

def generate_deepseek_response(session: list) -> str:
    try:
        formatted_prompt = deepseek_tokenizer.apply_chat_template(
            session,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = deepseek_pipeline(
            formatted_prompt,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.75,
            top_p=0.9,
            repetition_penalty=1.3,
            eos_token_id=deepseek_tokenizer.eos_token_id
        )
        # pdb.set_trace()
        content = outputs[0]['generated_text'].replace(formatted_prompt, '').strip()
        return validate_response(content)
    except Exception as e:
        print(f"DeepSeek Error: {str(e)}")
        return "[DeepSeek Response Failed]"

def validate_response(text: str) -> str:
    text = text.split('<|im_end|>')[0].strip()
    # if len(text.split()) > 75:
    #     text = ' '.join(text.split()[:75]) + '...'
    if text and text[-1] not in {'.', '!', '?'}:
        text += '.'
    return text

def conduct_debate(topic: str, traits: dict, rounds: int = 3) -> list:
    debate_log = []
    system_prompt = build_system_prompt(topic, traits)
    
    # Initialize sessions
    gpt_session = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Present your opening argument on {topic}"}
    ]
    
    deepseek_session = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Present your opening argument on {topic}"}
    ]

    # Generate opening statements
    deepseek_response = generate_deepseek_response(deepseek_session)
    debate_log.append(f"DeepSeek: {deepseek_response}")
    gpt_session.append({"role": "user", "content": f"previous argument: {deepseek_response}"})

    # Debate turns
    for _ in range(rounds):
        gpt_response = generate_gpt_response(gpt_session)
        debate_log.append(f"GPT-4: {gpt_response}")
        deepseek_session.append({"role": "user", "content": f"previous argument: {gpt_response}"})
        
        deepseek_response = generate_deepseek_response(deepseek_session)
        debate_log.append(f"DeepSeek: {deepseek_response}")
        gpt_session.append({"role": "user", "content": deepseek_response})

    return debate_log



def save_debates(output_file: str = "debate_logs_gptvsdeepseek.json"):
    # Load existing data if available
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Generate new debates
    new_debates = []
    for traits in TRAITS:
        for topic in TOPICS:
            print(f"Debating: {topic} with traits: {traits}")
            debate = conduct_debate(topic, traits)
            new_debates.append({
                "topic": topic,
                "traits": traits,
                "log": debate
            })

    # Combine existing and new data
    all_debates = existing_data + new_debates

    # Save updated data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_debates, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    save_debates()