from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain.chains import llm
from langchain.prompts import PromptTemplate
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'HuggingFaceH4/zephyr-7b-beta',
    task = 'text-generation'
)

prompt = PromptTemplate(
    input_variables=['topic'],
    template='suggest a catchy blg title about {topic}'
)

chain = llm(llm=llm,prompt=prompt)
topic = input("enter the topic")
output = chain.run(topic)

print(output)