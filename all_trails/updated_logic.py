import openai
import transformers
import torch
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found in .env file or environment variables")

class ModelManager:
    def __init__(self):
        self.deepseek = transformers.pipeline(
            "text-generation",
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            device_map="auto",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                # "trust_remote_code": True  # Keep this if required for the model
            }
        )
        
        self.system_prompt = (
            "As an expert story generator with advanced analytical capabilities, "
            "provide thorough, engaging, and well-structured responses. "
            "Maintain professional tone while ensuring accessibility."
        )

        self.discourse = []

    def _format_deepseek_prompt(self, history):
        """Structure prompts for DeepSeek with system message"""
        return f"System: {self.system_prompt}\n\n" + \
               "\n".join([f"Turn {i+1}: {msg['content']}" for i, msg in enumerate(history[-3:])])

    def generate_deepseek(self, history):
        """Generate response with context awareness"""
        try:
            formatted_prompt = self._format_deepseek_prompt(history)
            response = self.deepseek(
                formatted_prompt,
                max_new_tokens=150,
                temperature=0.65,
                top_p=0.9,
                repetition_penalty=1.1,
                num_return_sequences=1,
                truncation=True
            )
            
            generated = response[0]['generated_text'].replace(formatted_prompt, "").strip()
            return self._sanitize_response(generated)
            
        except Exception as e:
            print(f"DeepSeek Error: {str(e)}")
            return "[Model unavailable]"

    def generate_gpt(self, history):
        """Generate GPT response with proper message structure"""
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages += [{"role": "user" if i%2==0 else "assistant", "content": msg["content"]} 
                        for i, msg in enumerate(history[-4:])]
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=200,
                top_p=0.9
            )
            
            return self._sanitize_response(response.choices[0].message['content'])
            
        except Exception as e:
            print(f"GPT Error: {str(e)}")
            return "[API Error]"

    def _sanitize_response(self, text):
        """Clean and format model outputs"""
        text = text.strip()
        if not any(text.endswith(punct) for punct in ('.', '!', '?', '"', "'")):
            text += '.'
        return text

    def log_interaction(self, model, content):
        """Track conversation history"""
        entry = {"model": model, "content": content}
        self.discourse.append(entry)
        return entry

# Enhanced discussion facilitator
def conduct_discussion(topic, turns=4):
    manager = ModelManager()
    
    initial_prompt = (
        f"The discussion topic is: '{topic}'. "
        "Provide a comprehensive analysis of why this issue is significant, "
        "considering multiple perspectives and potential implications."
    )

    history = [{"model": "Facilitator", "content": initial_prompt}]
    
    for turn in range(turns):
        print(f"\n=== Turn {turn+1}/{turns} ===")
        
        # DeepSeek response
        deepseek_resp = manager.generate_deepseek(history)
        history.append(manager.log_interaction("DeepSeek-R1", deepseek_resp))
        print(f"\n[DeepSeek]\n{deepseek_resp}")
        
        # GPT-4 response
        gpt_prompt = f"Previous analysis: {deepseek_resp}\n\nProvide a counterpoint or complementary perspective."
        history.append({"model": "Facilitator", "content": gpt_prompt})
        
        gpt_resp = manager.generate_gpt(history)
        history.append(manager.log_interaction("GPT-4", gpt_resp))
        print(f"\n[GPT-4]\n{gpt_resp}")

    print("\n=== Final Discourse Summary ===")
    for entry in manager.discourse:
        print(f"\n[{entry['model']}]\n{entry['content']}")

# Execute with your climate change topic
if __name__ == "__main__":
    TOPIC = "Is climate change an immediate global crisis?"
    conduct_discussion(TOPIC, turns=4)