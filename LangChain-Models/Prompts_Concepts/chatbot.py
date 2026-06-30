from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
api_key = "HUGGINGFACEHUB_ACCESS_TOKEN" 
llm = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    huggingfacehub_api_token = api_key,
    task  = "text_generation"
)

model = ChatHuggingFace(llm=llm )

chat_history= []

while True:
    user_input = input("User:")
    chat_history.append(user_input)
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ",result.content)
   
print(chat_history)