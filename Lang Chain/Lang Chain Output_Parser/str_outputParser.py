from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

# 1st prompt --> detail report
template1 = PromptTemplate(
    template='write a detail report on {topic}',
    input_variables=['topic']
)

# 2nd prompt --> 5 line
template2 = PromptTemplate(
    template='write a 5 line summary on the following text.\n {text}',
    input_variables=['text']
)



prompt1 = template1.invoke({"topic":'black hole'})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result1.content})


result2 = model.invoke(prompt2)

print(result2.content)