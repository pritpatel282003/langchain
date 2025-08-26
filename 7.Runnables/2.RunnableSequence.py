from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

# Load env
load_dotenv()
huggingFaceToken = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Hugging Face model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=huggingFaceToken,
    temperature=0.0
)
prompt=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='explain the joke {text}',
    input_variables=['text']
)


model=ChatHuggingFace(llm=llm)
parser=StrOutputParser()
chain= RunnableSequence(prompt,model,parser,prompt2,model, parser)
result=chain.invoke({"topic":"man"})
print(result)
