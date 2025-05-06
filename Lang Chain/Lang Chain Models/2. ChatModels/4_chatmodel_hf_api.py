from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("top 5 best places to visit in uttarakhand ")

print(result.content)