from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(model="text-embeddings-3-large",dimensions=32)
result = model.embed_query("Lahore is the Capital of Pakistan")

print(str(result))
