from langchain_huggingface import ChatHuggingFace,HuggingfaceEndpoints
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
api_key = os.get_env("HUGGINGFACE_API_KEY")
load_dotenv()

llm = HuggingfaceEndpoints(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text_generation",
    huuggingfacehub_api_token = api_key
)

prompt = PromptTemplate(
    template="Generate 5 fact about topic{topic}",
    input_variables=['topic']
)

model = ChatHuggingFace(llm = llm) 

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)

