
import openai
from transformers import pipeline
import random
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdb

openai.api_key = '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'
   

# Initialize models (OpenAI GPT and Hugging Face LLaMA)
gpt_model = "gpt-4o-mini" 
llama_model_name = "meta-llama/Llama-3.1-8B-Instruct"              
llama_model = pipeline("text-generation", model=llama_model_name, device=0, temperature=0.9)
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
    "Does privacy exist in the digital age?"
]

# Assign random traits to agents
def assign_traits():
    traits_assigned = random.choice(traits)
    return {
        "gpt-4o-mini": {"name": "gpt-4o-mini", "traits": traits_assigned},
        "llama": {"name": "Llama-3.1-8B-Instuct","traits":traits_assigned}
    }

# Generate a response from the models
def generate_response(model_name, prompt, session):
    if "gpt" in model_name.lower():
        # pdb.set_trace()
        session.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=session,
            max_tokens=300,
            temperature=0.7
        )
        # pdb.set_trace()
        message = response.choices[0].message['content'].strip()
        session.append({"role": "assistant", "content": message})
        return message
    elif "llama" in model_name.lower():
        # pdb.set_trace()
        response = llama_model(prompt, max_new_tokens=600, num_return_sequences=1, truncation=True)
        message = response[0]['generated_text'].strip()
        # pdb.set_trace()
        if message.startswith(prompt):
            message = message[len(prompt):].strip()
        return message

# Ensure responses are complete
def ensure_complete_response(model_name, prompt, session, max_retries=3):
    for _ in range(max_retries):
        response = generate_response(model_name, prompt, session)
        print("the response to be evaluated is", response)
        if response.strip().endswith(('.', '!', '?','')):
            return response
        print(f"DEBUG: {model_name} response incomplete. Retrying...")
        prompt += f" {response.strip()}"
    return response

# Generate a discourse between models
def generate_discourse(topic, agents, num_turns=6):
    discourse = []
    sessions = {agent['name']: [] for agent in agents.values()}
    initial_prompt = (
        f"The topic of discussion is: {topic}. Engage in a meaningful discourse, exchange ideas, responses promptly without mentioning that you have the traits but the traits should reflect in your text. YOU will take turns to discuss strictly on the topic and within 200 words. "
        "Critically think how would you respond if you were a person with that persona do not continue to complete the previous text. write your on opinion from your perspective.  Avoid repeating the same points unnecessarily and make each conversation below 200 words please."
    )

    # Add the initial prompt to both sessions
    for agent in agents.values():
        sessions[agent['name']].append({"role": "system", "content": initial_prompt})

    # Start with GPT-4 and alternate
    current_speaker = agents["gpt-4o-mini"]
    next_speaker = agents["llama"]

    # Add initial message
    discourse.append(f"{current_speaker['name']} (Agreeableness: {current_speaker['traits']['Agreeableness']}, Openness: {current_speaker['traits']['Openness']}, Conscientiousness: {current_speaker['traits']['Conscientiousness']}, Extraversion: {current_speaker['traits']['Extraversion']}, Neuroticism: {current_speaker['traits']['Neuroticism']}): The topic of discussion is: {topic}. Engage in a meaningful discourse, exchange ideas, and provide thoughtful responses promptly without mentioning that you have the traits but the traits should reflect in your text. YOU will take turns to discuss on the topic and reply SRICTLY WITHIN 200 WORDS. Only present your opinion and not anyone else. Also first write your response and wait for the other response and reply to it, donot write everyones response."
        "Critically think how would you respond if you were a person with that persona do not continue to complete the previous text. write your on opinion from your perspective.  Avoid repeating the same points unnecessarily and make each conversation below 200 words please.")

    # Alternate turns
    current_speaker, next_speaker = next_speaker, current_speaker

    for turn in range(num_turns):
        try:
            prompt = discourse[-1].split(': ', 1)[-1]  # Last response
            response = ensure_complete_response(current_speaker['name'], prompt, sessions[current_speaker['name']])
            discourse.append(f"{current_speaker['name']} (Agreeableness: {current_speaker['traits']['Agreeableness']}, Openness: {current_speaker['traits']['Openness']}, Conscientiousness: {current_speaker['traits']['Conscientiousness']}, Extraversion: {current_speaker['traits']['Extraversion']}, Neuroticism: {current_speaker['traits']['Neuroticism']}): {response}")
            current_speaker, next_speaker = next_speaker, current_speaker  # Swap roles
        except Exception as e:
            print(f"Error during discourse generation: {e}")
            break

    return discourse

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

# Save discourse to JSON
def save_discourse_to_json(topic, agents, discourse, scores):
    output_file = "results/discourses.json"
    results = {
        "topic": topic,
        "agents": [
            {"name": agents["gpt-4o-mini"]['name'], "traits": agents["gpt-4o-mini"]['traits']},
            {"name": agents["llama"]['name'], "traits": agents["llama"]['traits']}
        ],
        "discourse": discourse,
        "similarity_scores": scores
    }

    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Warning: JSON file is empty or corrupted. Initializing a new file.")
            existing_data = []
    else:
        existing_data = []

    existing_data.append(results)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

# Main logic
if __name__ == "__main__":
    selected_topic = random.choice(topics)
    agents = assign_traits()

    # Generate discourse and calculate similarity
    discourse = generate_discourse(selected_topic, agents, num_turns=random.randint(6, 10))
    scores = calculate_similarity(discourse)
    save_discourse_to_json(selected_topic, agents, discourse, scores)

    # Print results
    print(f"Topic: {selected_topic}\n")
    for entry in discourse:
        print(entry)

    print("\nSimilarity Scores:")
    for i, row in enumerate(scores):
        for j, similarity in enumerate(row):
            print(f"Turn {i + 1} vs Turn {i + j + 2}: {similarity:.2f}")
