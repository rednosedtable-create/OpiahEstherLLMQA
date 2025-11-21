# app.py
import streamlit as st
import os
import re
import string
from groq import Groq

# Page config
st.set_page_config(page_title="LLM Q&A System", page_icon="🤖")

# Initialize Groq
client = Groq(api_key=st.secrets["GROQ_API_KEY"])  # We'll use secrets later


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return " ".join(tokens)


def get_answer(question):
    if not question.strip():
        return "Please enter a question."

    processed = preprocess_text(question)
    prompt = f"Answer concisely and accurately:\n{processed}"

    with st.spinner("Getting answer from LLM..."):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=512
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"API Error: {str(e)}"


# UI
st.title("🤖 LLM Question & Answer System")
st.markdown("Powered by **Groq + LLaMA 3**")

question = st.text_area("Enter your question:", height=100, placeholder="e.g., What is the capital of Japan?")

if st.button("Get Answer", type="primary"):
    if question:
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Original:** {question}")
            st.code(preprocess_text(question), language=None)
        with col2:
            answer = get_answer(question)
            st.success("**Answer:**")
            st.write(answer)
    else:
        st.warning("Please enter a question!")

st.markdown("---")
st.caption("Built for NLP Project 2 • 21 Nov 2025")
