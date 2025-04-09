import openai
import transformers
import torch
import random
import json
import os
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please configure it in a .env file or as an environment variable.")

# Initialize Hugging Face LLaMA model
llama_model = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device = 0 
)

# Define 10 combinations of Big Five personality traits for agents
traits = [
    {"Agreeableness": "High", "Openness": "Low", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "High"},
    # {"Agreeableness": "Low", "Openness": "High", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "Low"},
    # {"Agreeableness": "High", "Openness": "High", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "High"},
    # {"Agreeableness": "Low", "Openness": "Low", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "Low"},
    # {"Agreeableness": "High", "Openness": "High", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "Low"},
    # {"Agreeableness": "Low", "Openness": "High", "Conscientiousness": "High", "Extraversion": "High", "Neuroticism": "High"},
    # {"Agreeableness": "High", "Openness": "Low", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "Low"},
    # {"Agreeableness": "Low", "Openness": "Low", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "High"},
    # {"Agreeableness": "High", "Openness": "Low", "Conscientiousness": "High", "Extraversion": "High", "Neuroticism": "Low"},
    # {"Agreeableness": "Low", "Openness": "High", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "Low"}
]

# Topics for discourse
topics = [
    "Should artificial intelligence be used in military applications?",
    # "Is social media a threat to democracy?",
    # "Does capitalism cause environmental destruction?",
    # "Should governments regulate cryptocurrencies?",
    # "Is universal basic income a solution to automation?",
    # "Does privacy exist in the digital age?",
    # "Should the death penalty be abolished?",
    # "Is climate change an immediate global crisis?",
    # "Should vaccines be mandatory for everyone?",
    # "Is free speech under threat in the digital age?",
    # "Should billionaire wealth be capped?",
    # "Are electric cars truly sustainable?",
    # "Should governments control social media platforms?",
    # "Is space exploration worth the investment?",
    # "Should artificial intelligence have human-like rights?",
    # "Is universal healthcare a basic human right?",
    # "Does censorship protect society or stifle freedom?",
    # "Is online privacy a myth in the modern world?",
    # "Should education be completely free worldwide?",
    # "Is cryptocurrency the future of global finance?",
    # "Are social media influencers ruining modern culture?"
]



# Build system prompt
def build_system_prompt(topic):
    return (
        f"You are participating in a structured debate on the topic: '{topic}'. "
        "Your responses should reflect your assigned personality traits. "
        "Be concise, thoughtful, and under 50 words. Ensure your response is self-contained and complete."
    )

# Generate a response from GPT model
def generate_response_gpt(model_name, session):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=session,
        max_tokens=200,
        temperature=0.7
    )
    message = response.choices[0].message['content'].strip()
    if not message.endswith(('.', '!', '?')):
        message += '.'
    return message

# Generate a response from LLaMA model
def generate_response_llama(session):
    user_prompt = next((msg["content"] for msg in session if msg["role"] == "user"), "")
    response = llama_model(
        user_prompt,
        max_new_tokens=100,
        num_return_sequences=1,
        temperature=0.7,
        truncation=True
    )
    message = response[0]['generated_text'].strip()

    # Ensure response is concise and independent
    if user_prompt in message:
        message = message.replace(user_prompt, '').strip()
    if not message.endswith(('.', '!', '?')):
        message += '.'
    return message

# Ensure responses are complete
def ensure_complete_response(model_name, session):
    if model_name == "llama":
        return generate_response_llama(session)
    return generate_response_gpt(model_name, session)
# Generate a discourse between models

def generate_discourse(topic, traits_combination, num_turns=6):
    discourse = []
    system_prompt = build_system_prompt(topic)

    # Initialize sessions for all agents
    sessions = {
        "gpt-4o": [
            {"role": "system", "content": system_prompt}
        ],
        "llama": [
            {"role": "system", "content": system_prompt}
        ]
    }

    # Facilitator initiates the discussion
    facilitator_prompt = (
        f"The topic is: '{topic}'. Let's start the discussion. "
        f"LLaMA, you have the following personality traits: {traits_combination}. "
        "Please share your thoughts first."
    )
    # Facilitator initiates the discussion
    discourse.append(f"facilitator: {facilitator_prompt}")

    # LLaMA starts the discussion with the facilitator's prompt
    sessions["llama"].append({"role": "user", "content": facilitator_prompt})
    print("DEBUG: Initial session for LLaMA before response:", sessions["llama"])
    llama_response = ensure_complete_response("llama", sessions["llama"])
    print("DEBUG: LLaMA response:", llama_response)
    discourse.append(f"llama: {llama_response}")
    sessions["gpt-4o"].append({"role": "user", "content": llama_response})

    # Alternate between GPT-4o and LLaMA for the remaining turns
    current_speaker, next_speaker = "gpt-4o", "llama"

    for turn in range(1, num_turns):  # Starting from turn 1 as LLaMA already started
        try:
            session = sessions[current_speaker]
            # print(f"DEBUG: Session for {current_speaker} before response:", session)

            # Generate response from the current speaker
            response = ensure_complete_response(current_speaker, session)
            # print(f"DEBUG: {current_speaker} response:", response)

            # Append response to discourse
            discourse.append(f"{current_speaker}: {response}")

            # Update the next speaker's session with the current response
            if next_speaker == "llama" and turn == 1:  # First turn for LLaMA uses facilitator prompt
                pass  # Facilitator's prompt has already been handled above
            else:
                sessions[next_speaker].append({"role": "user", "content": response})

            # Swap roles for the next turn
            current_speaker, next_speaker = next_speaker, current_speaker

        except Exception as e:
            print(f"Error during discourse generation: {e}")
            break



# Generate and save discourse for all trait and topic combinations
def generate_and_save_discourse(traits, topics):
    all_results = []

    for traits_combination in traits:
        for topic in topics:
            discourse = generate_discourse(topic, traits_combination)
            all_results.append({"topic": topic, "traits": traits_combination, "discourse": discourse})

    # Save results to JSON
    output_file = "results/trail_1_discourses_combinations.json"
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Warning: JSON file is empty or corrupted. Initializing a new file.")
            existing_data = []
    else:
        existing_data = []

    existing_data.extend(all_results)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

# Main logic
if __name__ == "__main__":
    generate_and_save_discourse(traits, topics)
