from langchain_community.document_loaders import TextLoader
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

prompt=PromptTemplate(
    template='write a summary for the following {topic}',
    input_variables=['topic']
)

model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()
loader=TextLoader('requirements.txt',encoding='utf-8')
docs=loader.load()

chain = prompt | model | parser
print(chain.invoke({"topic":docs[0].page_content}))
