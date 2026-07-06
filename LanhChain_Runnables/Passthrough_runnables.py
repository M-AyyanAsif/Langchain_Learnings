from langchain_huggingface import HuggingFaceEndpoints, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough
import os

api_key = os.get_env("HUGGINGFACEHUB_API_KEY")

load_dotenv()

llm = HuggingFaceEndpoints(
    repo_id =  "openai-community/gpt2",
    task = "text_generation",
    huggingface_api_token  = api_key
)

template1 = PromptTemplate(
    template="Write a joke about the {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="Explain the following joke -{text}",
    input_variables= ['text']
)

parser = StrOutputParser()

model = ChatHuggingFace(llm = llm)

joke_gen_chain = RunnableSequence(template1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(template2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({'topic':'cricket'}))