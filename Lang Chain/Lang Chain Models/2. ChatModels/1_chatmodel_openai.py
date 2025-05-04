from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

result = model.invoke("what is the captital of india",temperature=0,max_completion_tokens=10)

# temp --> low value more determistic and predictable
# temp --> high value more random and creativea and diverse

print(result.content)

