import streamlit as st
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# ---------------------------
# 🔹 Load model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# 🔹 Prompt builder
# ---------------------------
def build_prompt(expenses_text):
    return f"""
You are a smart financial assistant.

User expenses:
{expenses_text}

Analyze and provide:
1. Categories of spending
2. Key insights
3. Saving suggestions

Keep it concise and clear.

Answer:
"""

# ---------------------------
# 🔹 Generate response
# ---------------------------
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

# ---------------------------
# 🔹 Streamlit UI
# ---------------------------
st.set_page_config(page_title="Budget Analyzer", page_icon="💰")

st.title("💰 Budget Analyzer (GPT-2)")
st.write("Enter your expenses and get AI-powered insights.")

# ---------------------------
# 🔹 Input options
# ---------------------------
option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

data = None

if option == "Manual Entry":
    user_input = st.text_area(
        "Enter expenses (example: Food: 5000, Rent: 15000)",
        height=150
    )

    if user_input:
        try:
            items = [x.strip() for x in user_input.split(",")]
            data = {}
            for item in items:
                key, value = item.split(":")
                data[key.strip()] = float(value.strip())

            df = pd.DataFrame(list(data.items()), columns=["Category", "Amount"])
        except:
            st.error("Invalid format. Use: Food: 5000, Rent: 15000")

elif option == "Upload CSV":
    file = st.file_uploader("Upload CSV with Category, Amount")

    if file:
        df = pd.read_csv(file)
        data = dict(zip(df["Category"], df["Amount"]))

# ---------------------------
# 🔹 Show chart
# ---------------------------
if 'df' in locals():
    st.subheader("📊 Expense Breakdown")
    st.bar_chart(df.set_index("Category"))

# ---------------------------
# 🔹 Analysis button
# ---------------------------
if st.button("🔍 Analyze Budget"):
    if data:
        expenses_text = "\n".join([f"{k}: {v}" for k, v in data.items()])
        prompt = build_prompt(expenses_text)

        with st.spinner("Analyzing..."):
            result = generate_analysis(prompt)

        st.subheader("🧠 AI Insights")
        st.write(result)

        # Simple scoring logic
        total = sum(data.values())
        max_category = max(data, key=data.get)

        st.subheader("📌 Quick Summary")
        st.write(f"Total Spending: ₹{total}")
        st.write(f"Highest Spending Category: {max_category}")

    else:
        st.warning("Please enter or upload data first.")
