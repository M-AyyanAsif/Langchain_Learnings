from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4", openai_api_key = api_key, temperature=1.1, max_completion_tokens=10)

model.invoke("What is the capital of Pakistan")
print(model)
