from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation'
)
model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()


class feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=feedback)



prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback in positive and negative {text} \n {format_instruction}',
    input_variables=['text'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser

prompt2 = PromptTemplate(
    template='write an appropriate resopnes ot this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write an appropriate resopnes ot this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] == 'positive',prompt2 | model | parser),
    (lambda x:x['sentiment'] == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

# result = classifier_chain.invoke({'text':'this is wornderfull mobile'}).sentiment

chain = classifier_chain | branch_chain

result = chain.invoke({"text":"this is s teriffier thing"})

print(result)
