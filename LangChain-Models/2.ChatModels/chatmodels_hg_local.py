from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()
llms = HuggingFacePipeline.from_model_id(
    model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task  = "text_generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens= 100
    )
)

model = HuggingFacePipeline(llm = llms)
result = model.invoke("What is the capital of Pakistan")

print(result.content)