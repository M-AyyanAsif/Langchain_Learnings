from langchain_huggingface import HuggingFaceEndpoints, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel
import os

api_key = os.get_env("HUGGINGFACEHUB_API_KEY")

load_dotenv()

llm = HuggingFaceEndpoints(
    repo_id =  "openai-community/gpt2",
    task = "text_generation",
    huggingface_api_token  = api_key
)

prompt1  = PromptTemplate(
    template="Generate a tweet about{topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= "Generate a linkdn posst about {topic} ",
    input_variables=['topic']
)

model = ChatHuggingFace(llm= llm)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableParallel(prompt1, model, parser),
    'linkdn': RunnableParallel(prompt2,model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkdn'])