from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

api_key = os.getenv("HUGGINGFACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text_generation",
    huggingfacehub_api_token= api_key
)

model  = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    temeplate = "Explain this {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Write a 5 line poem on the following text/n{text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic':'blackhole'})
print(result)
