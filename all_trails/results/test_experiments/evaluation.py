import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Synthetic Benchmark Data Generator
def generate_benchmark_data(num_samples=1000):
    np.random.seed(42)
    traits = ['Agreeableness', 'Openness', 'Conscientiousness', 'Extraversion', 'Neuroticism']
    
    benchmark = []
    for _ in range(num_samples):
        entry = {
            "topic": np.random.choice([
                "Social Media Impact", 
                "AI Ethics", 
                "Climate Change",
                "Mental Health"
            ]),
            "traits": {t: np.random.choice(["High", "Low"], p=[0.3, 0.7]) for t in traits},
            "discourse": [
                f"Turn {i+1}: Sample text about {np.random.choice(['ethics','technology','society'])}" 
                for i in range(np.random.randint(3, 7))
            ]
        }
        benchmark.append(entry)
    return benchmark

# 2. Discourse Data Processor
def process_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    processed = []
    for entry in data:
        traits = entry['traits']
        text = ' '.join(entry['discourse'])
        processed.append({
            'topic': entry['topic'],
            'traits': traits,
            'text': text,
            'trait_vector': [
                1 if traits['Agreeableness'] == 'High' else 0,
                1 if traits['Openness'] == 'High' else 0,
                1 if traits['Conscientiousness'] == 'High' else 0,
                1 if traits['Extraversion'] == 'High' else 0,
                1 if traits['Neuroticism'] == 'High' else 0
            ]
        })
    return pd.DataFrame(processed)

# 3. Comparative Analyzer
class DiscourseBenchmarker:
    def __init__(self, user_data, benchmark_data):
        self.user_df = process_data(user_data)
        self.bench_df = pd.DataFrame(generate_benchmark_data())
        
        # Align text features
        self.vectorizer = TfidfVectorizer(max_features=500)
        self._fit_vectorizers()
    
    def _fit_vectorizers(self):
        combined_text = pd.concat([self.user_df['text'], self.bench_df['text']])
        self.vectorizer.fit(combined_text)
    
    def compare_trait_distributions(self):
        results = {}
        for trait in ['Agreeableness', 'Openness', 'Conscientiousness', 'Extraversion', 'Neuroticism']:
            user = self.user_df['traits'].apply(lambda x: x[trait]).map({'High':1, 'Low':0})
            bench = self.bench_df['traits'].apply(lambda x: x[trait]).map({'High':1, 'Low':0})
            
            # Kolmogorov-Smirnov test
            stat, p_value = ks_2samp(user, bench)
            results[trait] = {
                'user_mean': user.mean(),
                'bench_mean': bench.mean(),
                'ks_statistic': stat,
                'p_value': p_value
            }
        return results
    
    def compare_text_patterns(self):
        user_vectors = self.vectorizer.transform(self.user_df['text'])
        bench_vectors = self.vectorizer.transform(self.bench_df['text'])
        
        similarity = cosine_similarity(user_vectors, bench_vectors)
        return {
            'mean_similarity': similarity.mean(),
            'max_similarity': similarity.max(),
            'min_similarity': similarity.min()
        }
    
    def generate_report(self):
        report = {
            'trait_comparison': self.compare_trait_distributions(),
            'text_similarity': self.compare_text_patterns()
        }
        
        # Visualization
        self._plot_distributions()
        return report
    
    def _plot_distributions(self):
        plt.figure(figsize=(15, 10))
        for idx, trait in enumerate(['Agreeableness', 'Openness', 'Conscientiousness', 'Extraversion', 'Neuroticism'], 1):
            plt.subplot(2, 3, idx)
            user = self.user_df['traits'].apply(lambda x: x[trait])
            bench = self.bench_df['traits'].apply(lambda x: x[trait])
            
            pd.DataFrame({
                'Your Data': user.value_counts(normalize=True),
                'Benchmark': bench.value_counts(normalize=True)
            }).plot(kind='bar', title=trait)
        
        plt.tight_layout()
        plt.savefig('trait_distribution_comparison.png')
        plt.close()

# Usage Example
if __name__ == "__main__":
    # Generate synthetic benchmark data
    with open('synthetic_benchmark.json', 'w') as f:
        json.dump(generate_benchmark_data(), f)
    
    # Initialize benchmarker with your data
    analyzer = DiscourseBenchmarker('discourse.json', 'synthetic_benchmark.json')
    
    # Generate full report
    report = analyzer.generate_report()
    
    # Save results
    with open('benchmark_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("""
    Benchmarking complete!
    - Trait comparison: benchmark_report.json
    - Visualizations: trait_distribution_comparison.png
    """)