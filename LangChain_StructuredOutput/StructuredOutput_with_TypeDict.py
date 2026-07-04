from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from typing import TypedDict,Annotated

load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_KEY")
llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    temperature = 0.5,
    max_new_tokens= 100,
    huggingfacehub_api_token = "HuggingFaceH4/zephyr-7b-beta"
    )

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary: Annotated[str,"A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review"]

Structured_model = model.with_structured_output(Review)

result = Structured_model.invoke("""Mobile hardware today is powerful, with fast processors, high RAM, and smooth displays enabling heavy apps and gaming.
Battery life and camera quality have improved significantly, making phones more reliable for daily and professional use.
However,heating issues and high prices in flagship devices are still common drawbacks.""")

print(result)
