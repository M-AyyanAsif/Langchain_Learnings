from langchain_huggingface import ChatHuggingFace , HuggingfaceEndpoints
from langchain_core.runnables import RunnableParallel, RunnableBranch,RunnableLambda
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
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

model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description=" Give the sentiment of the following feedback")

parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template="Classify the feedback of the give statement into positive or negative\n {feedback}, \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instruction()}
)

prompt2 = PromptTemplate(
    template=" Write an appropiate response to the positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template=" Write an appropiate response to the negtive feedback \n {feedback}",
    input_variables=['feedback']
)

classifier_chain = prompt1 | model | parser
result = claasifeir_chain.invoke({'feedback':'this is a very terrible smartphone'}).sentiment

print(result)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment = "positive", prompt2 | model | parser),
    (lambda x : x.sentiment = "negative", prompt3 | model | parser),
    lambda x: "could not find sentiment")



final_chain = claasifeir_chain | branch_chain

result = final_chain.invoke({'feedback': 'this is a terrible phone'})
print(result)