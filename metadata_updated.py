import json
import nltk
import numpy as np
import pandas as pd
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu
# nltk.data.path.append('/media/data_4tb/pranav/llms_gen_eval/')

# nltk.download("punkt")
# nltk.data.path.append('/home/24492114/nltk_data/tokenizers')
# 📌 List of JSON Files to Process
json_files={
    "GPT-4o_vs_GPT-4o-mini": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gptvs_gpt_discorse/final_discourse_gptvsgpt_combinations.json",  # Replace with actual paths
    "LLaMA-3_vs_GPT-4": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_llama_discorse/final_discourse_ptvsllama_combinations.json",
    "DeepSeek_vs_LLaMA": "/media/data_4tb/pranav/llms_gen_eval/parse_json/final_gpt_vs_deepseek_discorse/final_discorse_gptvsdeepseek.json"
}
# 📌 Store Results
all_results = []

# 📌 Process Each File
for model_comparison, json_file in json_files.items():
    print(f"🔍 Processing {model_comparison}...")

    # Load JSON File
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 📌 Initialize Variables
    total_dialogues = 0
    total_words = 0
    total_sentences = 0
    total_utterances = 0
    bleu_scores = []
    
    speaker_word_counts = {}
    speaker_sentence_counts = {}

    # 📌 Process Each Topic
    for entry in data:
        topic = entry["topic"]
        discourse = entry["discourse"]
        
        total_dialogues += 1  # Each topic is a dialogue

        for utterance in discourse:
            speaker, text = utterance.split(": ", 1)  # Extract Speaker and Text
            
            # Count Sentences & Words
            num_sentences = len(sent_tokenize(text))
            num_words = len(word_tokenize(text))
            
            total_sentences += num_sentences
            total_words += num_words
            total_utterances += 1

            # Store Speaker Stats
            if speaker not in speaker_word_counts:
                speaker_word_counts[speaker] = []
                speaker_sentence_counts[speaker] = []

            speaker_word_counts[speaker].append(num_words)
            speaker_sentence_counts[speaker].append(num_sentences)
        
        # 📌 Compute BLEU Score (Text Similarity Between Agents)
        for i in range(1, len(discourse)):
            reference = word_tokenize(discourse[i - 1].split(": ", 1)[1])  # Previous Utterance
            candidate = word_tokenize(discourse[i].split(": ", 1)[1])  # Current Utterance
            bleu_score = sentence_bleu([reference], candidate)
            bleu_scores.append(bleu_score)

    # 📌 Compute Final Averages
    avg_dialogues = total_dialogues / len(data)
    avg_words_per_dialogue = total_words / total_dialogues
    avg_utterance_length = total_words / total_utterances
    avg_bleu_score = np.mean(bleu_scores) if bleu_scores else 0

    # 📌 Compute Per-Speaker Stats
    avg_words_per_speaker = {spk: np.mean(words) for spk, words in speaker_word_counts.items()}
    avg_sentences_per_speaker = {spk: np.mean(sentences) for spk, sentences in speaker_sentence_counts.items()}

    # 📌 Store Results
    all_results.append({
        "Model Comparison": model_comparison,
        "Total Dialogues": total_dialogues,
        "Total Words": total_words,
        "Total Sentences": total_sentences,
        "Total Utterances": total_utterances,
        "Avg Dialogues per Topic": avg_dialogues,
        "Avg Words per Dialogue": avg_words_per_dialogue,
        "Avg Utterance Length": avg_utterance_length,
        "Avg BLEU Score": avg_bleu_score,
        "Avg Words per Speaker": avg_words_per_speaker,
        "Avg Sentences per Speaker": avg_sentences_per_speaker
    })

# 📌 Save Results to CSV
df_results = pd.DataFrame(all_results)
df_results.to_csv("metadata_analysis_results.csv", index=False)

print("✅ Analysis Completed! Results saved to 'metadata_analysis_results.csv'.")
