from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm = HuggingFaceEmbeddings(
    repo_id = 'HuggingFaceH4/zephyr-7b-beta',
    task = 'text-generation'
)
model1 = ChatHuggingFace(llm = llm)
model2 = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'generate a short and simple note form the following {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = 'generate 5 short question answer from the following {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = 'merge the provided notes and quizes into a single document {notes} and {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()


paralled_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = paralled_chain | merge_chain

text = ''

result = chain.invoke({'text':text})
print(result)
