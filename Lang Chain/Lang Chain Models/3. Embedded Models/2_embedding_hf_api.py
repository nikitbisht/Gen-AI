from langchain_huggingface import  HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    repo_id = 'sentence-transformers/all-MiniLM-L6-v2',
    dimensions=32
)

result = embedding.embed_query("delhi is the cpaital of india")

print(str(result))