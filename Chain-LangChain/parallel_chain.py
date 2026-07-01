from langchain_huggingface import ChatHuggingFace , HuggingfaceEndpoints
from langchain_core.runnables import RunnableParallel
from langchain_anthropic import ChatAnthropic
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

model1 = ChatHuggingFace(llm = llm)
model2 = ChatAnthropic(model = "claude-3-sonnet-20250219")

prompt1 = PromptTemplate(
    template="Generate short and simple note from the follwing text\n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = "Generate 5 short question answers from the following text\n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and the quizes into a single document\n -> notes   {notes} and quiz -> {quiz}",
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

paralllel_chains = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
}
)

merge_chain = prompt3 | model1 | parser

chain = paralllel_chains | merge_chain

text = """gkjghsjdgb skbkdfbjlsdb salbvksbkdlgb dskbsksbdlkjgbsda bsdkvbksbvkjsab foehfbgvdag vvkjbvjbv
bjvbkv kjbaskvd bvkjbkvbabkda saboawehto iowgbl;gwq wbjsabljbasd dbfjsdblfkjsabd  sdjbbkjdbkjsd jbsbksdabt 
"""

result = chain.invoke({'text' : text})
print(result)

