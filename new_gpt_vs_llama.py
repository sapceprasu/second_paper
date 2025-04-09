import openai
import transformers
import torch
import json
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import pdb


# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found in .env or environment variables")

## Quantization settings for LLaMA-3.1-8B
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
        f"The Personality traits that needs to be reflected in your text are: {json.dumps(traits)}\n"
        "Rules:\n"
        "- Maintain this personality traits (DONOT EXPLIITLY CMENTION IN TEXT) at all time during your conversation\n"
        "- Keep responses under 50 words\n"
        "- Maintain your personality consistently\n"
        "- Address previous points directly\n"
        "- Use natural conversational English<|eot_id|>"
    )

def generate_gpt_response(session: list) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=session,
            temperature=0.7,
            max_tokens=150
        )
        content = response.choices[0].message['content'].strip()
        return validate_response(content)
    except Exception as e:
        print(f"GPT Error: {str(e)}")
        return "[GPT Response Failed]"

def generate_llama_response(session: list) -> str:
    try:
        formatted_prompt = llama_tokenizer.apply_chat_template(
            session,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = llama_pipeline(
            formatted_prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=llama_tokenizer.eos_token_id,
            pad_token_id=llama_tokenizer.eos_token_id
        )

        #
        pdb.set_trace()
        
        content = outputs[0]['generated_text'].replace(formatted_prompt, '').strip()
        return validate_response(content)
    except Exception as e:
        print(f"LLaMA Error: {str(e)}")
        return "[LLaMA Response Failed]"

def validate_response(text: str) -> str:
    text = text.split('<|eot_id|>')[0].strip()
    if len(text.split()) > 75:
        text = ' '.join(text.split()[:75]) + '...'
    if text and text[-1] not in {'.', '!', '?'}:
        text += '.'
    return text

def conduct_debate(topic: str, traits: dict, rounds: int = 4) -> list:
    debate_log = []
    system_prompt = build_system_prompt(topic, traits)
    
    # Initialize sessions with LLaMA-3 format
    gpt_session = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Present your opening argument on the topic {topic}."}
    ]
    
    llama_session = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Present your opening argument on the topic {topic}."}
    ]

    # Generate opening statements
    llama_response = generate_llama_response(llama_session)
    debate_log.append(f"LLaMA-3: {llama_response}")
    gpt_session.append({"role": "user", "content": llama_response})
    llama_session.append({"role": "assistant", "content": llama_response})

    # Debate turns
    for turn in range(rounds):
        # GPT response
        if turn<=8:
            gpt_response = generate_gpt_response(gpt_session)
            debate_log.append(f"GPT-4: {gpt_response}")
            llama_session.append({"role": "user", "content": gpt_response})
            gpt_session.append({"role": "assistant", "content": gpt_response})
            
            # LLaMA response
            llama_response = generate_llama_response(llama_session)
            debate_log.append(f"LLaMA-3: {llama_response}")
            gpt_session.append({"role": "user", "content": llama_response})
            llama_session.append({"role": "assistant", "content": llama_response})

    return debate_log

def save_debates(output_file: str = "final_discourse_ptvsllama_combinations.json"):
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
                "discourse": debate
            })
            pdb.set_trace()

        # pdb.set_trace()

    # Combine existing and new data
    all_debates = existing_data + new_debates

    # Save updated data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_debates, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    save_debates()