from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)
# chat_history = []
messages = [
    SystemMessage(content = 'You are a helpful assistance')
]
while(1):
    user_input = input("You:- ")
    messages.append(HumanMessage(content = user_input))

    if user_input == "exit":
        break

    result = model.invoke(messages)
    messages.append(AIMessage(content = result.content))

    print("AI: ",result.content)

print(messages)