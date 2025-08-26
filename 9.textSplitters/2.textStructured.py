from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('D:\GEN_AI\langchain\Prit Sanjiv Patel_AIML.pdf')
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0
)
chunks=splitter.split_documents(docs) 
print(len(chunks))
print(chunks[0].page_content)