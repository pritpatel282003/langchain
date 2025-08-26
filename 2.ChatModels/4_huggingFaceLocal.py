from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

os.environ['HF_HOME']='D:/huggingface_cache'
llm=HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    ),
    huggingfacehub_api_token=api_key
    

)

model=ChatHuggingFace(llm=llm)
result=model.invoke("what is the capital of india")
print(result.content)