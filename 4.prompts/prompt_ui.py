import os
from dotenv import load_dotenv
import streamlit as st
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Load env
load_dotenv()
os.environ['HF_HOME'] = 'D:/huggingface_cache'

# Detect device
device = 0 if torch.cuda.is_available() else -1

# Initialize model (Flan-T5 handles instructions better than Bart)
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    pipeline_kwargs=dict(
        temperature=0.3,
        device=device
    )
)

# ---------------- STREAMLIT UI ----------------
st.title("📄 Research Paper Summarizer with LangChain")

user_input = st.text_area("Paste your paper text here:", height=250)

# Summarization styles
options = {
    "Short Summary": "Summarize the following text in 2–3 sentences.",
    "Detailed Summary": "Write a detailed summary covering all key points.",
    "Bullet Points": "Summarize the text into concise bullet points.",
    "Technical Summary": "Summarize the paper for a technical researcher.",
    "Layman’s Summary": "Summarize the paper in simple terms for the general public."
}

style = st.selectbox("Choose summarization style:", list(options.keys()))

# LangChain PromptTemplate
template = """You are a helpful research assistant.
{instruction}

Text:
{text}
"""

prompt = PromptTemplate(
    input_variables=["instruction", "text"],
    template=template
)

if st.button("Summarize"):
    if user_input.strip():
        # Fill prompt dynamically
        formatted_prompt = prompt.format(
            instruction=options[style],
            text=user_input
        )

        # Run through model
        summary = llm.invoke(formatted_prompt)

        st.subheader("🔎 Summary")
        st.write(summary)
    else:
        st.warning("⚠️ Please paste some text to summarize.")
