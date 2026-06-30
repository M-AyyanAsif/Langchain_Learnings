from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate
import os
api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

model = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token= api_key,
    Temperature = 0.7,
    max_new_tokens = 512
)

load_dotenv()

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name",["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])
style_input = st.selectbox("Select Explanation Type",["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])
lenght_input = st.selectbox("Selct Explantion Type",["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

Prompt = PromptTemplate.invoke({
    "paper_input":paper_input,
    "style_input":style_input,
    "lenght_input":lenght_input
})

if  st.button("Summarize"):
    result = model.invoke(Prompt)
    st.write(result.content)