from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    # repo_id= 'HuggingFaceH4/zephyr-7b-beta',
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



# chain using runnable sequential

chain = RunnableSequence(prompt1,model,parser,prompt2,model,parser)

result = chain.invoke({'topic':'pakistan'})

print(result)
