from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('D:\GEN_AI\langchain\Prit Sanjiv Patel_AIML.pdf')
docs=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=99,
    chunk_overlap=25,
    separator=''
)
result=splitter.split_documents(docs)
for i in docs:
    print(i.page_content)