import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
Analyze text segments from two anonymous debaters (Person One and Person Two) for:
1. Big Five Inventory (BFI) traits (High/Low for each dimension)
2. Consistency with typical behavior for those traits (Yes/No)

For each person, return:
{
    "predicted_bfi": {
        "Agreeableness": "High/Low",
        "Openness": "High/Low",
        "Conscientiousness": "High/Low",
        "Extraversion": "High/Low",
        "Neuroticism": "High/Low"
    },
    "consistent_with_traits": "Yes/No"
}
"""

def analyze_bfi(text, persona):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze {persona}'s text:\n{text}"}
            ],
            temperature=0.2,
            max_tokens=500
        )
        print("the results are:",response.choices[0].message['content'])
        return json.loads(response.choices[0].message['content'])
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None

def process_discourse(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        # Separate messages by persona
        person_one = []
        person_two = []
        
        for msg in entry['discourse']:
            if "DeepSeek:" in msg:
                person_one.append(msg.split(":", 1)[1].strip())
            elif "GPT-4:" in msg:
                person_two.append(msg.split(":", 1)[1].strip())
        
        # Analyze each persona
        entry['analysis'] = {
            "Person_One": analyze_bfi("\n".join(person_one), "Person One"),
            "Person_Two": analyze_bfi("\n".join(person_two), "Person Two")
        }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    process_discourse("/media/data_4tb/pranav/llms_gen_eval/final_discorse_gptvsdeepseek.json","bfi_analysis_gpt40mini_gptvsdeepseek.json")