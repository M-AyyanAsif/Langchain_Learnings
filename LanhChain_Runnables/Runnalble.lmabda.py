from langchain_huggingface import HuggingFaceEndpoints, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda
import os

api_key = os.get_env("HUGGINGFACEHUB_API_KEY")

load_dotenv()

llm = HuggingFaceEndpoints(
    repo_id =  "openai-community/gpt2",
    task = "text_generation",
    huggingface_api_token  = api_key
)

def word_count(text):
    print(len(text.split()))

prompt1 = PromptTemplate(
    template="Write a prompt about the {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

model = ChatHuggingFace(llm =llm)

joke_gene_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
    })

final_chain = RunnableSequence(joke_gene_chain, parallel_chain)

print(final_chain.invoke({'topic','AI'}))

result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

print(result)

