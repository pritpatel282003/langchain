from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load env
load_dotenv()
huggingFaceToken = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Hugging Face model (chat-compatible)
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="conversational",
    huggingfacehub_api_token=huggingFaceToken,
    temperature=0.0
)


# Wrap into Chat model
model = ChatHuggingFace(llm=llm)

# Prompt template
prompt = PromptTemplate(
    template="Answer the following question:\n{question}\nFrom this text:\n{text}",
    input_variables=["question", "text"]
)

# Load webpage content
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Black_hole")
docs = loader.load()

# Parser
parser = StrOutputParser()

# Chain: prompt -> model -> parser
chain = prompt | model | parser

# Run query
result = chain.invoke({
    "question": "Summarize black hole in 10 words",
    "text": docs[0].page_content[:32000]
})

print(result)
