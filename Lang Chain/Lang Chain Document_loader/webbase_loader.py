from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'HuggingFaceH4/zephyr-7b-beta',
    task = 'text-generation',
    max_new_tokens= 20
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'answer the following question \n{question} from the following text - \n{text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/realme-p3x-5g-midnight-blue-128-gb/p/itmab5a4b09b6ccc?pid=MOBH8VGVTPGFVHEX&lid=LSTMOBH8VGVTPGFVHEX1Z6RGO&marketplace=FLIPKART&sattr[]=color&sattr[]=ram&st=color'

loader = WebBaseLoader(url) #also pass list of urls

docs = loader.load()

# print(docs[0].page_content)
# print(len(docs))

chain = prompt | model | parser

result =  chain.invoke({'text':docs[0].page_content,'question':'tell me all the specification of the product'})

print(result)