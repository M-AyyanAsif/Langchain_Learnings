from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

api_key = os.getenv("HUGGINGFACE_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text_generation",
    huggingfacehub_api_token= api_key
)

model  = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

tempelate= PromptTemplate(
    template="write a five line summary of /n{format_instruction}",
    input_types=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = tempelate|model|parser 
result = chain.invoke({} )

print(result)

