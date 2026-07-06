from langchain_huggingface import HuggingFaceEndpoints, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough , RunnableLambda, RunnableBranch
import os

api_key = os.get_env("HUGGINGFACEHUB_API_KEY")

load_dotenv()

llm = HuggingFaceEndpoints(
    repo_id =  "openai-community/gpt2",
    task = "text_generation",
    huggingface_api_token  = api_key
)


prompt1 = PromptTemplate(
    template="Write a detalied report about the {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()
model = ChatHuggingFace(llm =llm)

prompt2 = PromptTemplate(
    template="Write the summary of the following\n {text}",
    input_variables= ['text']
)

report_gen_llm = prompt1|model|parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2|model|parser)
)

final_chain = report_gen_llm | branch_chain

print(final_chain.invoke({'topic':'Russia_vs_Ukraine'}))
