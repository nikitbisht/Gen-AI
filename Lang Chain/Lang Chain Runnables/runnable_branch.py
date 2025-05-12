from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    max_new_tokens = 20
)
model = ChatHuggingFace(llm=llm)



prompt1 = PromptTemplate(
    template = 'write detail report on {topic}',
    input_variable=['topic']
)
prompt2 = PromptTemplate(
    template = 'summarize the following text {text}',
    input_variables=['text']
)
parser = StrOutputParser()

def word_count(text):
    return len(text.split())



# chain using runnable lambda

gen_chain = RunnableSequence(prompt1,model,parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split()) > 500, RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(gen_chain , branch_chain)
result = final_chain.invoke({'topic':'pakistan'})

print(result)
