from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#chat template
chat_template = ChatPromptTemplate([
    ('system',"you are a helpfull customer support agent"),
    MessagesPlaceholder(variable_name='chat_history'),      #used to load previous chat_history from database
    ('human', "{query}")
])

chat_history= []

#load chat history 
with open('Lang Chain Prompts/chat_history.txt') as f :
    chat_history.extend(f.readlines())

print(chat_history)


#create our prompt

prompt = chat_template.invoke({
    'chat_history':chat_history,
    'query':'where is my refund'
})


print("\n")
print(prompt)
