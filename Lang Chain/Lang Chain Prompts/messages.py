from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = '',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

messages = [
    SystemMessage(content = 'You are a helpfull assistance'),
    HumanMessage(content = 'Tell me about Langchain')
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))
