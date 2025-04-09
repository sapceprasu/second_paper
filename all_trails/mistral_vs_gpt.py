import openai
import transformers
import torch
import random
import json
import os
from dotenv import load_dotenv

# Load OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please configure it in a .env file or as an environment variable.")

# Initialize Hugging Face DeepSeek model
deepseek_model = transformers.pipeline(
    "text-generation",
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",  # Automatically map the model across available GPUs
)


# Define combinations of Big Five personality traits for agents
traits = [
    # {"Agreeableness": "High", "Openness": "Low", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "High"},
    # {"Agreeableness": "Low", "Openness": "High", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "Low"},
    # {"Agreeableness": "High", "Openness": "High", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "High"},
    # {"Agreeableness": "Low", "Openness": "Low", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "Low"},
    {"Agreeableness": "High", "Openness": "High", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "Low"}
]

# Topics for discourse
topics = [
    "Should artificial intelligence be used in military applications?",
    # "Is social media a threat to democracy?",
    # "Does capitalism cause environmental destruction?",
    # "Should governments regulate cryptocurrencies?",
    # "Is universal basic income a solution to automation?"
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

# Ensure single instance loading and reuse in all interactions
def generate_response_deepseek(session):
    """Generate a response using DeepSeek model."""
    user_prompt = next((msg["content"] for msg in session if msg["role"] == "user"), "")
    print("DEBUG: Session content passed to DeepSeek:", session)
    print("DEBUG: User prompt for DeepSeek:", user_prompt)

    response = deepseek_model(
        user_prompt,
        max_new_tokens=100,
        num_return_sequences=1,
        temperature=0.7,
        truncation=True
    )
    generated_text = response[0]['generated_text'].strip()

    # Remove user prompt if repeated in the response
    if user_prompt in generated_text:
        generated_text = generated_text.replace(user_prompt, "").strip()

    # Ensure complete response ends with punctuation
    if not generated_text.endswith(('.', '!', '?')):
        generated_text += '.'

    return generated_text

# Ensure responses are complete
def ensure_complete_response(model_name, session):
    if model_name == "deepseek":
        return generate_response_deepseek(session)
    return generate_response_gpt(model_name, session)

# Generate a discourse between models
def generate_discourse(topic, traits_combination, num_turns=6):
    discourse = []
    system_prompt = build_system_prompt(topic)

    # Initialize sessions for all agents
    sessions = {
        "gpt-4o": [],
        "deepseek": []
    }

    # Facilitator initiates the discussion
    facilitator_prompt = (
        f"The topic is: '{topic}'. Let's start the discussion. "
        f"DeepSeek, you have the following personality traits: {traits_combination}. "
        "Please share your thoughts first."
    )
    discourse.append(f"facilitator: {facilitator_prompt}")

    # DeepSeek starts the discussion
    sessions["deepseek"] = [
        {"role": "user", "content": facilitator_prompt},
        {"role": "system", "content": system_prompt}  # Placeholder for system updates
    ]
    print("DEBUG: Initial session for DeepSeek before response:", sessions["deepseek"])
    deepseek_response = ensure_complete_response("deepseek", sessions["deepseek"])
    print("DEBUG: DeepSeek response:", deepseek_response)
    discourse.append(f"deepseek: {deepseek_response}")

    # Update GPT session with DeepSeek's response
    sessions["gpt-4o"] = [
        {"role": "user", "content": deepseek_response},
        {"role": "system", "content": facilitator_prompt}
    ]

    # Alternate between GPT-4o and DeepSeek for the remaining turns
    current_speaker, next_speaker = "gpt-4o", "deepseek"

    for turn in range(num_turns - 1):
        try:
            session = sessions[current_speaker]
            print(f"DEBUG: Session for {current_speaker} before response:", session)

            # Generate response from the current speaker
            response = ensure_complete_response(current_speaker, session)
            print(f"DEBUG: {current_speaker} response:", response)

            # Append response to discourse
            discourse.append(f"{current_speaker}: {response}")

            # Update next speaker's session
            sessions[next_speaker] = [
                {"role": "user", "content": response},
                {"role": "system", "content": session[0]["content"]}
            ]

            # Swap roles for the next turn
            current_speaker, next_speaker = next_speaker, current_speaker

        except Exception as e:
            print(f"Error during discourse generation: {e}")
            break

    return discourse

# Generate and save discourse for all trait and topic combinations
def generate_and_save_discourse(traits, topics):
    all_results = []

    for traits_combination in traits:
        for topic in topics:
            discourse = generate_discourse(topic, traits_combination)
            all_results.append({"topic": topic, "traits": traits_combination, "discourse": discourse})

    # Save results to JSON
    output_file = "results/discourses_combinations.json"
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
