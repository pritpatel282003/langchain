from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

prompt1=PromptTemplate(
    template='Generate top 5 athlets of {name}',
    input_variables=['name']
)

prompt2=PromptTemplate(
    template='Rank them from first to last and just give their names {result}',
    input_variables=['result']
)

model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()
chain=prompt1 | model | parser | prompt2 | model | parser
result=chain.invoke({"name":"cricket"})
print(result)
chain.get_graph().print_ascii()