import openai
import random
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader

# Load OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please configure it in a .env file or as an environment variable.")

# Define random Big Five personality traits for agents
traits = [
    {"Agreeableness": "High", "Openness": "Low", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "High"},
    {"Agreeableness": "Low", "Openness": "High", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "Low"},
    {"Agreeableness": "High", "Openness": "High", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "High"},
    {"Agreeableness": "Low", "Openness": "Low", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "Low"},
    {"Agreeableness": "High", "Openness": "High", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "Low"},
    {"Agreeableness": "Low", "Openness": "High", "Conscientiousness": "High", "Extraversion": "High", "Neuroticism": "High"}
]

# Topics for discourse
topics = [
    "Should artificial intelligence be used in military applications?",
    "Is social media a threat to democracy?",
    "Does capitalism cause environmental destruction?",
    "Should governments regulate cryptocurrencies?",
    "Is universal basic income a solution to automation?",
    "Does privacy exist in the digital age?",
    "Should the death penalty be abolished?",
    "Is climate change an immediate global crisis?",
    "Should vaccines be mandatory for everyone?",
    "Is free speech under threat in the digital age?",
    "Should billionaire wealth be capped?",
    "Are electric cars truly sustainable?",
    "Should governments control social media platforms?",
    "Is space exploration worth the investment?",
    "Should artificial intelligence have human-like rights?",
    "Is universal healthcare a basic human right?",
    "Does censorship protect society or stifle freedom?",
    "Is online privacy a myth in the modern world?",
    "Should education be completely free worldwide?",
    "Is cryptocurrency the future of global finance?",
    "Are social media influencers ruining modern culture?"
]

# Dataset for batching inputs
class DiscourseDataset(Dataset):
    def __init__(self, topics, num_conversations):
        self.data = [{"topic": random.choice(topics), "turn": i} for i in range(num_conversations)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Build system prompt
def build_system_prompt(topic):
    return (
        f"You are participating in a structured debate on the topic: '{topic}'. "
        "Your responses should reflect your assigned personality traits. "
        "Be concise, thoughtful, and under 200 words. Ensure your response is self-contained and complete."
    )

# Assign random traits to agents
def assign_traits():
    traits_assigned = random.choice(traits)
    return {
        "facilitator": {"name": "facilitator", "traits": None},
        "gpt-4o": {"name": "gpt-4o", "traits": traits_assigned},
        "llama": {"name": "llama", "traits": traits_assigned}
    }

# Generate a response from GPT model
def generate_response_gpt(model_name, session):
    """Generate a response from GPT models."""
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=session,
        max_tokens=200,  # Adjust token limit for response length
        temperature=0.7  # Balance between randomness and relevance
    )
    message = response.choices[0].message['content'].strip()

    # Ensure the response ends with punctuation
    if not message.endswith(('.', '!', '?')):
        message += '.'

    return message

# Ensure responses are complete
def ensure_complete_response(model_name, session):
    """Generates a complete response."""
    if model_name == "llama":
        user_prompt = next((msg["content"] for msg in session if msg["role"] == "user"), "")
        return f"LLaMA Response to: {user_prompt}"
    return generate_response_gpt(model_name, session)

# Generate a discourse between models
def generate_discourse(topic, agents, num_turns=6):
    """Generates a structured debate between the agents."""
    discourse = []
    system_prompt = build_system_prompt(topic)

    # Initialize sessions for all agents
    sessions = {
        agent['name']: [
            {"role": "system", "content": system_prompt}
        ] for agent in agents.values()
    }

    # Facilitator initiates the discussion
    facilitator_prompt = f"The topic is: '{topic}'. Let's start the discussion. LLaMA, please share your thoughts first."
    sessions["facilitator"].append({"role": "user", "content": facilitator_prompt})
    discourse.append(f"facilitator: {facilitator_prompt}")

    # LLaMA starts the discussion
    llama_response = ensure_complete_response("llama", sessions["facilitator"])
    discourse.append(f"llama: {llama_response}")
    sessions["gpt-4o"].append({"role": "user", "content": llama_response})

    # Alternate between GPT-4o and LLaMA for the remaining turns
    current_speaker, next_speaker = agents["gpt-4o"], agents["llama"]

    for turn in range(num_turns - 1):
        try:
            session = sessions[current_speaker["name"]]

            # Generate response
            response = ensure_complete_response(current_speaker["name"], session)

            # Append response to discourse and update session
            discourse.append(f"{current_speaker['name']} (Traits: {current_speaker['traits']}): {response}")
            sessions[next_speaker["name"].append({"role": "user", "content": response})]

            # Swap roles for the next turn
            current_speaker, next_speaker = next_speaker, current_speaker
        except Exception as e:
            print(f"Error during discourse generation: {e}")
            break

    return discourse

# Generate and save discourse using a DataLoader
def generate_and_save_discourse(topics, agents, num_conversations=10):
    dataset = DiscourseDataset(topics, num_conversations)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    all_results = []
    for batch in dataloader:
        # Validate batch structure
        if isinstance(batch, list) and all(isinstance(item, dict) for item in batch):
            for item in batch:
                topic = item["topic"]
                discourse = generate_discourse(topic, agents)
                scores = calculate_similarity(discourse)
                all_results.append({"topic": topic, "discourse": discourse, "similarity_scores": scores})
        else:
            print(f"DEBUG: Skipping malformed batch: {batch}")

    # Save results to JSON
    output_file = "results/discourses_batched.json"
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

# Calculate similarity scores
def calculate_similarity(discourse):
    vectorizer = TfidfVectorizer()
    messages = [entry.split(': ', 1)[-1] for entry in discourse if ': ' in entry]
    tfidf_matrix = vectorizer.fit_transform(messages)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    compressed_scores = []
    for i in range(len(messages)):
        compressed_scores.append([
            round(similarity_matrix[i, j], 3) for j in range(i + 1, len(messages))
        ])
    return compressed_scores

# Main logic
if __name__ == "__main__":
    agents = assign_traits()
    generate_and_save_discourse(topics, agents, num_conversations=20)
