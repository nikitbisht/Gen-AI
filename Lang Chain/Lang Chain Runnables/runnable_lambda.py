from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    max_new_tokens = 20
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variable=['topic']
)

parser = StrOutputParser()

def word_count(text):
    return len(text.split())



# chain using runnable lambda

joke_gen_chain = RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_cnt': RunnableLambda(word_count) #RunnableLambda(lambda x:len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain , parallel_chain)
result = final_chain.invoke({'topic':'pakistan'})

print(result)
