from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage


chat_template = ChatPromptTemplate([
    #using tupple
    ('system','you are a helpfull {domain} expert'),
    ('human', 'explain in simple term what is {topic}')
])

prompt = chat_template.invoke({
    'domain': 'cricket',
    'topic': 'Dustra'
})

print(prompt)