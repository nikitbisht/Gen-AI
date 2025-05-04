from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model= 'text-embedding-3-large',dimensions=32)         #default small model 1536  3072 for large model
document = [
    "delhi is capital of india",
    "dehradun is a cpaital of uk"
]
result = embedding.embed_query(document)

print(str(result))