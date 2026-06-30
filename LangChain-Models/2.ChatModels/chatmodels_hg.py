from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
HF_Token = ("HUGGGINGFACEHUB_ACCESS_TOKEN")

llm =HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    huggingfacehub_api_token = HF_Token,
    task  = "text_generation"
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is the capital of Pakistan")
print(result)