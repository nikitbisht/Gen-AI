from langchain_community.document_loaders import TextLoader
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
    template = 'write a summary for the following poem \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()


loader = TextLoader('cricket.txt',encoding='utf-8')

docs = loader.load()

print(type(docs))
print(len(docs))
# print(docs[0].page_content)

chain = prompt | model | parser

result =  chain.invoke({'poem':docs[0].page_content})

print(result)