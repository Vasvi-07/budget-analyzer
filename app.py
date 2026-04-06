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
        max_length=300,  # Increased slightly for better detail
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Budget Analyzer", page_icon="💰")
st.title("💰 AI Budget Analyzer (GPT-2)")
st.write("Enter your monthly expenses below to get AI-powered financial advice.")

st.subheader("Monthly Expenses")
# Creating a fill-in-the-blank style layout using columns
col1, col2 = st.columns(2)

with col1:
    food = st.number_input("🍔 Food & Groceries", min_value=0, step=100)
    rent = st.number_input("🏠 Rent / Hostel", min_value=0, step=500)
    transport = st.number_input("🚗 Transport / Fuel", min_value=0, step=50)

with col2:
    shopping = st.number_input("🛍️ Shopping", min_value=0, step=100)
    subs = st.number_input("📺 Subscriptions (Netflix, etc.)", min_value=0, step=10)
    other = st.number_input("✨ Other Expenses", min_value=0, step=50)

# Calculate total automatically
total_spent = food + rent + transport + shopping + subs + other

# Format the data for the AI prompt
structured_expenses = f"""
Food: {food}
Rent: {rent}
Transport: {transport}
Shopping: {shopping}
Subscriptions: {subs}
Other: {other}
TOTAL SPENT: {total_spent}
"""

if st.button("Analyze Budget"):
    if total_spent == 0:
        st.warning("Please enter at least one expense value.")
    else:
        with st.spinner("AI is analyzing your spending..."):
            prompt = build_prompt(structured_expenses)
            raw_result = generate_analysis(prompt)
            
            # Cleaning up the output: removing the prompt from the response
            analysis_only = raw_result.split("Answer:")[-1].strip()

            st.divider()
            st.subheader("📊 AI Analysis")
            st.info(f"**Total Monthly Spend:** {total_spent}")
            st.write(analysis_only)
