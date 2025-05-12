from langchain_huggingface import HuggingFacePipeline
from huggingface_hub import login

llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0,
        "max_new_tokens": 100,
    }
)

result = llm.invoke("who is tha prime minister of india")
print(result)
