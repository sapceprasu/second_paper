import json
import spacy
import pandas as pd
from textblob import TextBlob
from nltk.translate.bleu_score import sentence_bleu
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define file paths for different discourse comparisons
json_files = {
 "GPT-4o_vs_GPT-4o-mini": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/final_discourse_gptvsgpt_combinations.json",  # Replace with actual paths
    "LLaMA-3_vs_GPT-4": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/final_discourse_ptvsllama_combinations.json",
    "DeepSeek_vs_LLaMA": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/final_discorse_gptvsdeepseek.json"
}

# Define logical structure keywords
logical_keywords = ["if", "then", "because", "therefore"]

# Store results
metadata_results = []

def analyze_discourse(discourse):
    """Compute sentence-level and word-level statistics for a discourse."""
    num_sentences = 0
    total_words = 0
    total_utterances = len(discourse)
    assertions = 0
    questions = 0
    logical_count = 0

    for utterance in discourse:
        doc = nlp(utterance)
        
        # Convert doc.sents to a list
        sentences = list(doc.sents)
        num_sentences += len(sentences)
        
        # Count words
        words = [token.text for token in doc if not token.is_punct]
        total_words += len(words)
        
        # Count assertions vs. questions
        if "?" in utterance:
            questions += 1
        else:
            assertions += 1
        
        # Count logical structures
        logical_count += sum(1 for word in words if word.lower() in logical_keywords)
    
    avg_words_per_sentence = total_words / num_sentences if num_sentences else 0
    avg_utterance_length = total_words / total_utterances if total_utterances else 0

    return {
        "total_utterances": total_utterances,
        "total_sentences": num_sentences,
        "total_words": total_words,
        "avg_words_per_sentence": avg_words_per_sentence,
        "avg_utterance_length": avg_utterance_length,
        "assertions": assertions,
        "questions": questions,
        "logical_structure_count": logical_count
    }

# Process each JSON file
for dataset_name, file_path in json_files.items():
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping {dataset_name} - File not found: {file_path}")
        continue

    print(f"🔍 Processing {dataset_name}...")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_meta = {
        "dataset": dataset_name,
        "total_utterances": 0,
        "total_sentences": 0,
        "total_words": 0,
        "assertions": 0,
        "questions": 0,
        "logical_structure_count": 0,
        "total_dialogues": len(data),
    }

    for entry in data:
        discourse_stats = analyze_discourse(entry["discourse"])

        # Aggregate values
        total_meta["total_utterances"] += discourse_stats["total_utterances"]
        total_meta["total_sentences"] += discourse_stats["total_sentences"]
        total_meta["total_words"] += discourse_stats["total_words"]
        total_meta["assertions"] += discourse_stats["assertions"]
        total_meta["questions"] += discourse_stats["questions"]
        total_meta["logical_structure_count"] += discourse_stats["logical_structure_count"]

    # Compute final averages
    total_meta["avg_words_per_sentence"] = total_meta["total_words"] / total_meta["total_sentences"] if total_meta["total_sentences"] else 0
    total_meta["avg_utterance_length"] = total_meta["total_words"] / total_meta["total_utterances"] if total_meta["total_utterances"] else 0

    # Store results
    metadata_results.append(total_meta)

# Save to CSV
output_csv = "metadata_analysis_results.csv"
df_metadata = pd.DataFrame(metadata_results)
df_metadata.to_csv(output_csv, index=False)

print(f"✅ Metadata analysis completed! Results saved to {output_csv}")