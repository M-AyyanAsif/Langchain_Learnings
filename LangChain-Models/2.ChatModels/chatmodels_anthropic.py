from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("Anthropic_API_KEY")

model = ChatAnthropic(model="claude-3-5-sonet-20241022", anthropic_api_key= api_key)
result = model.invoke("What is the capital of Pakistan")
print(result.content)
