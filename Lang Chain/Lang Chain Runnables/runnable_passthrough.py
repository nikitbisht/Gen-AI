from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    max_new_tokens = 20
)
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variable=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'explain this joke {text}',
    input_variable = ['text']
)



# chain using runnable passthrough

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explain_joke': RunnableSequence(prompt2,model,parser)
})

chain = joke_gen_chain | parallel_chain
result = chain.invoke({'topic':'pakistan'})

print(result)
