from langchain_huggingface import ChatHuggingFace , HuggingfaceEndpoints
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

prompt1 = PromptTemplate(
    template="Generate a detailed prompt {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate more fact from the following {text}",
    input_variables=['text']
)
model = ChatHuggingFace(llm = llm) 

parser = StrOutputParser()

chain = prompt1| model| parser| prompt2| model| parser

result  = chain.invoke({'topic':'blackhole'})
print(result)
