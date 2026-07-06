from langchain_huggingface import HuggingFaceEndpoints, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
import os

api_key = os.get_env("HUGGINGFACEHUB_API_KEY")

load_dotenv()

llm = HuggingFaceEndpoints(
    repo_id =  "openai-community/gpt2",
    task = "text_generation",
    huggingface_api_token  = api_key
)

template = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables= ['topic']
)

model = ChatHuggingFace(llm = llm)

parser= StrOutputParser()

chain = RunnableSequence(template ,model ,parser)

print(chain.invoke({'topic':'AI'}))

