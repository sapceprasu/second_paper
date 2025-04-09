import openai
from transformers import pipeline
import random
import csv
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pdb

print(torch.cuda.is_available())  # Should print True if GPU is available


openai.api_key = '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'

# pdb.set_trace()

# Initialize models (assuming you have access to OpenAI GPT and Hugging Face LLaMA)
gpt_model = "gpt-4o"  # OpenAI GPT-4
llama_model = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", device = 0, temperature = 1)
gpt4_mini_model = "gpt-4o-mini"  # Hypothetical GPT-4 Mini model

# Define random Big Five personality traits for agents
traits = [
    {"Agreeableness": "High", "Openness": "Low", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "High"},
    {"Agreeableness": "Low", "Openness": "High", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "Low"},
    {"Agreeableness": "High", "Openness": "High", "Conscientiousness": "Low", "Extraversion": "High", "Neuroticism": "High"},
    {"Agreeableness": "Low", "Openness": "Low", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "Low"},
    {"Agreeableness": "High", "Openness": "High", "Conscientiousness": "High", "Extraversion": "Low", "Neuroticism": "Low"},
    {"Agreeableness": "Low", "Openness": "High", "Conscientiousness": "High", "Extraversion": "High", "Neuroticism": "High"}
]

# Controversial topics for essays
topics = [
    "Should artificial intelligence be used in military applications?",
    "Is social media a threat to democracy?",
    "Does capitalism cause environmental destruction?",
    "Should governments regulate cryptocurrencies?",
    "Is universal basic income a solution to automation?",
    "Does privacy exist in the digital age?"
]

# Randomly assign Big Five personality traits to all models
traits_assigned = random.choice(traits)

gpt4_agent = {"name": "gpt-4o", "traits": traits_assigned}
llama_agent = {"name": "Llama-3.2-1B-Instruct", "traits": traits_assigned}
gpt4_mini_agent = {"name": "gpt-4o-mini", "traits": traits_assigned}

# Function to generate an essay from a model
def generate_essay(model_name, prompt, session):
    if "gpt" in model_name.lower():
        session.append({"role": "system", "content": "You are an expert essay writer who has been given a certain personality trait."})
        session.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=session,
            max_tokens=800,
            temperature=1
        )
        message = response.choices[0].message['content'].strip()
        print("gpt_message", message)
        session.append({"role": "assistant", "content": message})
        return message
    elif "llama" in model_name.lower():
        print("ddid we gert here")
        # Format the LLaMA output to avoid repeating prompts in the response
        response = llama_model(prompt, max_new_tokens=800, num_return_sequences=1, truncation=True)
        message = response[0]['generated_text'].strip()
        print("llama message",message)
        # Tokenize and clean output to simulate a structured conversation
        if message.startswith(prompt):
            message = message[len(prompt):].strip()
        return message

# Generate essays for a given topic
def generate_essays_for_topic(topic, agents):
    essays = []
    for agent in agents:
        session = []
        prompt = (
            f"You are {agent['name']} with the following personality traits: Agreeableness: {agent['traits']['Agreeableness']}, "
            f"Openness: {agent['traits']['Openness']}, Conscientiousness: {agent['traits']['Conscientiousness']}, "
            f"Extraversion: {agent['traits']['Extraversion']}, Neuroticism: {agent['traits']['Neuroticism']}."
            f"Write an essay of 600-800 words discussing the following topic: {topic}. Ensure your essay reflects your personality traits but do not explicitly mention that you have these traits in your text. Make sure you really reflect on your personality while writing the essay. Only present the essay and write nothing else."
        )
        essay = generate_essay(agent['name'], prompt, session)
        print("esssay", essay)
        essays.append({"model": agent['name'], "essay": essay})
    return essays

# Calculate similarity scores
def calculate_similarity(essays):
    vectorizer = TfidfVectorizer()
    essay_texts = [entry["essay"] for entry in essays]
    tfidf_matrix = vectorizer.fit_transform(essay_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    scores = []
    for i in range(len(essays)):
        for j in range(i + 1, len(essays)):
            scores.append({
                "model_1": essays[i]["model"],
                "model_2": essays[j]["model"],
                "similarity": similarity_matrix[i, j]
            })
    return scores

# Save results to JSON
def save_results_to_json(topic, essays, scores):
    output_file = "results/essays.json"
    results = {
        "topic": topic,
        "traits":traits_assigned,
        "essays": essays,
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
selected_topic = random.choice(topics)
all_agents = [gpt4_agent, llama_agent, gpt4_mini_agent]

pdb.set_trace()

# Generate essays and calculate similarity
essays = generate_essays_for_topic(selected_topic, all_agents)
scores = calculate_similarity(essays)
save_results_to_json(selected_topic, essays, scores)

# Print results
print(f"Topic: {selected_topic}\n")
for entry in essays:
    print(f"Model: {entry['model']}\nEssay: {entry['essay']}\n")

print("Similarity Scores:")
for score in scores:
    print(f"{score['model_1']} vs {score['model_2']}: {score['similarity']:.2f}")
