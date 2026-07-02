from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

api_key = os.get_env("HUGGINGFACEHUB_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="LiquidAI/LFM2.5-350M",
    task="text_generation",
    huggingfacehub_api_token= api_key
)

model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template="wWrite a five line summary of the following poem -\n{poem}",
    input_variables=['poem']
)
parser= StrOutputParser()
loader = TextLoader("australia.txt")
docs = loader.load()
print(type(docs))
print(len(docs))
print(docs[0].metadata)
chain = prompt | model | parser
print(chain.invoke({'poem':docs[0].page_content}))

