# LLM_QA_CLI.py
import os
import re
import string
from groq import Groq  # Using Groq (fast & free tier available)

# Initialize Groq client (you need a free API key from https://console.groq.com)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return " ".join(tokens)

def get_llm_response(question):
    processed = preprocess_text(question)
    prompt = f"You are a helpful assistant. Answer this question concisely:\n{processed}"

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",  # Fast & free on Groq
            temperature=0.7,
            max_tokens=512
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("=== LLM Question & Answer System (CLI) ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Ask your question: ").strip()
        if question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue

        print(f"\nOriginal : {question}")
        processed = preprocess_text(question)
        print(f"Processed: {processed}")
        print("Thinking...", end="\n\n")

        answer = get_llm_response(question)
        print(f"Answer   : {answer}\n{'-'*50}\n")

if __name__ == "__main__":
    # Set your Groq API key in environment or replace below
    if not os.getenv("GROQ_API_KEY"):
        print("Please set GROQ_API_KEY environment variable!")
        print("Get free key: https://console.groq.com/keys")
    else:
        main()
