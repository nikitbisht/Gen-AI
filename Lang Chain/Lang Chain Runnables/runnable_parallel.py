from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= 'HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    max_new_tokens = 20
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = 'generate a linkedin post about {topic}',
    input_variable=['topic']
)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = 'generate a tweet about {topic}',
    input_variable = ['topic']
)

# chain using runnable sequential


paralled_chain = RunnableParallel({
    'linkedin': RunnableSequence(prompt1,model,parser),
    'tweet': RunnableSequence(prompt2,model,parser)
})


result = paralled_chain.invoke({'topic':'Gen-Ai Future'})

print(result)
