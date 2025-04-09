import openai
import transformers
import torch
import random
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
import pdb

# pdb.set_trace()


# Load OpenAI API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please configure it in a .env file or as an environment variable.")

# Initialize Hugging Face LLaMA model
llama_model = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

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
        "Be concise, thoughtful, and under 50 words."
    )

# Assign random traits to agents
def assign_traits():
    traits_assigned = random.choice(traits)
    return {
        "gpt-4o": {"name": "gpt-4o", "traits": traits_assigned},
        "llama": {"name": "Llama-3.1-8B-Instruct", "traits": traits_assigned}
    }

# Generate a response from LLaMA model
def generate_response_llama(session):
    """Generate a response from LLaMA using the latest user prompt."""
    # Extract the latest user prompt
    user_prompt = next((msg["content"] for msg in session if msg["role"] == "user"), "")
    # pdb.set_trace()
    # Debug input prompt
    print(f"DEBUG: LLaMA input prompt: {user_prompt}")

    # Generate response from LLaMA
    response = llama_model(
        user_prompt,
        max_new_tokens=120,  # Increased token limit for complete responses
        num_return_sequences=1,
        temperature=0.8,  # More diverse output
        truncation=True,
    )

    # Extract and process the generated text
    message = response[0]['generated_text'].strip()

    # Remove repeated content and handle empty responses
    if message.startswith(user_prompt):
        message = message[len(user_prompt):].strip()
    if not message:
        message = "I couldn't generate a response. Could you clarify?"

    # Add a final punctuation if missing
    if not message.endswith(('.', '!', '?')):
        message += '.'

    # Debug output
    print(f"DEBUG: LLaMA output: {message}")

    return message


# Ensure responses are complete
def ensure_complete_response(model_name, session, topic, max_retries=3):
    """Ensures a complete response with retries if needed."""
    for retry in range(max_retries):
        if "llama" in model_name.lower():
            # LLaMA Response
            response = generate_response_llama(session)
        else:  # GPT-4
            # GPT-4 Response
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=session,
                max_tokens=300,
                temperature=0.7
            ).choices[0].message['content'].strip()

        # Validate if the response ends with punctuation
        if response.strip().endswith(('.', '!', '?','')):
            return response

        # Log debug info and retry
        print(f"DEBUG: {model_name} response incomplete. Retrying ({retry + 1}/{max_retries})...")
        session.append({"role": "user", "content": response.strip()})

    # If retries are exhausted, return the best attempt
    return response

# Generate a discourse between models
def generate_discourse(topic, agents, num_turns=6):
    """Generates a structured debate between the agents."""
    discourse = []
    system_prompt = build_system_prompt(topic)

    # Initialize sessions for both agents
    sessions = {
        agent['name']: [
            {"role": "system", "content": system_prompt}
        ] for agent in agents.values()
    }

    # Initial user prompt
    initial_message = f"The topic is: '{topic}'. Let's start the discussion."
    sessions["gpt-4o"].append({"role": "user", "content": initial_message})
    discourse.append(f"gpt-4o: {initial_message}")

    # Alternate between agents for the specified number of turns
    current_speaker, next_speaker = agents["llama"], agents["gpt-4o"]

    for turn in range(num_turns):
        try:
            session = sessions[current_speaker["name"]]

            # Log debug info
            print(f"DEBUG: {current_speaker['name']} session before response: {session}")

            # Generate response
            response = ensure_complete_response(current_speaker["name"], session, topic)

            # Append response to discourse and update session
            discourse.append(f"{current_speaker['name']} (Traits: {current_speaker['traits']}): {response}")
            sessions[next_speaker["name"]].append({"role": "user", "content": response})

            # Swap roles for the next turn
            current_speaker, next_speaker = next_speaker, current_speaker
        except Exception as e:
            print(f"Error during discourse generation: {e}")
            break

    return discourse

# Generate responses for batches
def generate_batch_responses(batch, agents, num_turns=6):
    """Processes a batch of topics for generating discourse."""
    results = []
    for item in batch:
        topic = item["topic"]
        discourse = generate_discourse(topic, agents, num_turns)
        scores = calculate_similarity(discourse)
        results.append({"topic": topic, "discourse": discourse, "similarity_scores": scores})
    return results

def collate_fn(batch):
    """Ensures that DataLoader returns a list of dictionaries."""
    return [dict(item) for item in batch]



def generate_and_save_discourse(topics, agents, num_conversations=10):
    dataset = DiscourseDataset(topics, num_conversations)

    pdb.set_trace()

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)  # Use custom collate_fn

    all_results = []
    for batch in dataloader:
        # Validate batch structure
        if isinstance(batch, list) and all(isinstance(item, dict) for item in batch):
            batch_results = generate_batch_responses(batch, agents, num_turns=6)
            all_results.extend(batch_results)
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
