from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = HuggingFaceEmbeddings(
    repo_id = '',
    task = 'text-generation'
)
model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = 'generate 5 intersting facts about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'cricket'})

print(result)
