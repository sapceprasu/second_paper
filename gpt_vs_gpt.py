import openai
import json
import os
from dotenv import load_dotenv
import pdb
# Load OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please configure it in a .env file or as an environment variable.")



# pdb.set_trace()
 #Load the JSON data
with open("debate_topics_traits.json", "r") as file:
    data = json.load(file)

TOPICS = data["debate_topics"]
TRAITS = data["personality_traits"]


MODELS = ["gpt-4o", "gpt-4o-mini"]  # The two models to debate
OUTPUT_FILE = "final_discourse_gptvsgpt_combinations.json"

def build_system_prompt(topic, traits):
    return (
        f"You are participating in a structured debate on: '{topic}'\n"
        "Your responses should reflect these personality traits:\n"
        f"- Agreeableness: {traits['Agreeableness']}\n"
        f"- Openness: {traits['Openness']}\n"
        f"- Conscientiousness: {traits['Conscientiousness']}\n"
        f"- Extraversion: {traits['Extraversion']}\n"
        f"- Neuroticism: {traits['Neuroticism']}\n\n"
        "Rules:\n"
        "- Maintain this personality traits (DONOT EXPLIITLY MENTION IN TEXT) at all time during your conversation\n"
        "- Keep responses under 50 words\n"
        "- Maintain your personality consistently\n"
        "- Address previous arguments directly but do not repeat what the other speaker said.\n"
        "- End with proper punctuation"
    )

def generate_response(model, messages):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            max_tokens=150,
            temperature=0.9
        )
        content = response.choices[0].message['content'].strip()
        if not content.endswith(('.', '!', '?')):
            content += '.'
        return content
    except Exception as e:
        print(f"Error generating response from {model}: {str(e)}")
        return f"[{model} response failed]"

def generate_discourse(topic, traits):
    discourse = []
    system_prompt = build_system_prompt(topic, traits)
    pdb.set_trace()
    # Initialize both models with the same traits
    model1, model2 = MODELS[0], MODELS[1]
    
    # Create session histories
    sessions = {
        model1: [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The debate topic is: {topic}. Present your opening argument."}
        ],
        model2: [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The debate topic is: {topic}. Present your opening argument."}
        ]
    }

    # Generate opening arguments
    for model in MODELS:
        response = generate_response(model, sessions[model])
        discourse.append(f"{model}: {response}")
        # Update both sessions
        sessions[model1].append({"role": "assistant", "content": response})
        sessions[model2].append({"role": "user", "content": response})

    # Generate rebuttals
    for turn in range(6):  # 4 rounds of rebuttals
        current_model = MODELS[turn % 2]
        response = generate_response(current_model, sessions[current_model])
        discourse.append(f"{current_model}: {response}")
        # Update both sessions
        sessions[MODELS[0]].append({"role": "assistant" if MODELS[0] == current_model else "user", "content": response})
        sessions[MODELS[1]].append({"role": "assistant" if MODELS[1] == current_model else "user", "content": response})

    return discourse

def generate_and_save_discourse():
    # Load existing data
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Generate new debates
    new_data = []
    for traits in TRAITS:
        for topic in TOPICS:
            print(f"Processing: {topic}")
            discourse = generate_discourse(topic, traits)
            new_data.append({
                "topic": topic,
                "traits": traits,
                "discourse": discourse
            })

    # Combine and save

    all_data = existing_data + new_data
    # os.msakedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)


        # with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # json.dump(all_data,  with open(OUTPUT_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    

if __name__ == "__main__":
    generate_and_save_discourse()