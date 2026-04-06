import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# -----------------------------
# Load Model (runs once)
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt(expenses_text):
    return f"""
You are a financial assistant helping a student manage money.

User expenses:
{expenses_text}

Analyze and provide:
1. Spending categories
2. Key insights
3. Suggestions to save money

Answer:
"""

# -----------------------------
# Text Generation
# -----------------------------
def generate_analysis(prompt):

    inputs = tokenizer.encode(prompt, return_tensors="pt")

    outputs = model.generate(
        inputs,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💰 AI Budget Analyzer (GPT-2)")
st.write("Enter your monthly expenses and get AI financial advice.")

user_input = st.text_area(
    "Enter your expenses:",
    placeholder="""Food: 5000
Rent: 15000
Shopping: 4000
Transport: 2000
Subscriptions: 1200"""
)

if st.button("Analyze Budget"):

    if user_input.strip() == "":
        st.warning("Please enter expenses first.")
    else:
        prompt = build_prompt(user_input)
        result = generate_analysis(prompt)

        st.subheader("📊 AI Analysis")
        st.write(result)
