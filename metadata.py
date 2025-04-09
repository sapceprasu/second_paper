import json
import pandas as pd
import re
from textblob import TextBlob
from collections import Counter
import os

# File paths - Update these with actual file paths
input_files = {
    "GPT-4o_vs_GPT-4o-mini": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/final_discourse_gptvsgpt_combinations.json",  # Replace with actual paths
    "LLaMA-3_vs_GPT-4": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/final_discourse_ptvsllama_combinations.json",
    "DeepSeek_vs_LLaMA": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/final_discorse_gptvsdeepseek.json"
}

# 📌 Define output CSV
output_file = "average_metadata_analysis.csv"

# 📌 Define logical connectors
logical_connectors = ["if", "then", "because", "therefore", "thus", "hence"]

# 📌 Function to analyze discourse metadata
def analyze_discourse(data):
    total_p1_texts = []
    total_p2_texts = []

    for entry in data:
        discourses = entry["discourse"]
        
        p1_texts = [d for d in discourses if "gpt-4o" in d.lower() or "llama-3" in d.lower() or "deepseek" in d.lower()]
        p2_texts = [d for d in discourses if "gpt-4o-mini" in d.lower() or "gpt-4" in d.lower()]
        
        total_p1_texts.extend(p1_texts)
        total_p2_texts.extend(p2_texts)

    # 📌 Helper function for metadata extraction
    def extract_metadata(texts):
        num_assertions = 0
        num_questions = 0
        logical_count = 0
        sentiment_scores = []
        total_words = 0
        total_sentences = 0
        total_chars = 0
        
        for text in texts:
            # Count assertions and questions
            num_questions += len(re.findall(r"\?", text))
            num_assertions += len(re.findall(r"\.", text)) - num_questions
            
            # Count logical structures
            logical_count += sum(text.lower().count(conn) for conn in logical_connectors)
            
            # Sentiment analysis
            sentiment_scores.append(TextBlob(text).sentiment.polarity)
            
            # Count words and sentences
            words = text.split()
            total_words += len(words)
            total_sentences += max(len(re.split(r'[.!?]', text)), 1)  # Avoid division by zero
            
            # Character count
            total_chars += len(text)
        
        num_utterances = len(texts) if texts else 1
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        avg_words_per_utterance = total_words / num_utterances
        avg_words_per_sentence = total_words / total_sentences if total_sentences else 0
        avg_chars_per_utterance = total_chars / num_utterances
        
        return {
            "Assertions": num_assertions / num_utterances,
            "Questions": num_questions / num_utterances,
            "Logical_Structures": logical_count / num_utterances,
            "Avg_Sentiment": avg_sentiment,
            "Avg_Words_Per_Utterance": avg_words_per_utterance,
            "Avg_Words_Per_Sentence": avg_words_per_sentence,
            "Avg_Chars_Per_Utterance": avg_chars_per_utterance,
            "Total_Words": total_words,
            "Total_Sentences": total_sentences
        }
    
    # 📌 Compute averages for P1 and P2
    p1_stats = extract_metadata(total_p1_texts)
    p2_stats = extract_metadata(total_p2_texts)

    return {
        "Assertions_P1": p1_stats["Assertions"], "Assertions_P2": p2_stats["Assertions"],
        "Questions_P1": p1_stats["Questions"], "Questions_P2": p2_stats["Questions"],
        "Logical_Structures_P1": p1_stats["Logical_Structures"], "Logical_Structures_P2": p2_stats["Logical_Structures"],
        "Avg_Sentiment_P1": p1_stats["Avg_Sentiment"], "Avg_Sentiment_P2": p2_stats["Avg_Sentiment"],
        "Avg_Words_Per_Utterance_P1": p1_stats["Avg_Words_Per_Utterance"], "Avg_Words_Per_Utterance_P2": p2_stats["Avg_Words_Per_Utterance"],
        "Avg_Words_Per_Sentence_P1": p1_stats["Avg_Words_Per_Sentence"], "Avg_Words_Per_Sentence_P2": p2_stats["Avg_Words_Per_Sentence"],
        "Avg_Chars_Per_Utterance_P1": p1_stats["Avg_Chars_Per_Utterance"], "Avg_Chars_Per_Utterance_P2": p2_stats["Avg_Chars_Per_Utterance"],
        "Total_Words_P1": p1_stats["Total_Words"], "Total_Words_P2": p2_stats["Total_Words"],
        "Total_Sentences_P1": p1_stats["Total_Sentences"], "Total_Sentences_P2": p2_stats["Total_Sentences"]
    }

# 📌 Process all datasets
final_results = []

for dataset_name, file_path in input_files.items():
    if os.path.exists(file_path):
        print(f"📊 Processing {dataset_name}...")
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            dataset_results = analyze_discourse(data)
            dataset_results["Dataset"] = dataset_name
            final_results.append(dataset_results)
    else:
        print(f"🚨 File not found: {file_path}")

# 📌 Convert results to DataFrame and save
df_results = pd.DataFrame(final_results)
df_results.to_csv(output_file, index=False)

print(f"✅ Metadata analysis completed and saved to '{output_file}'!")
