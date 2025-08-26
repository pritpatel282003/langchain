from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
docs=[
    'hi',
    'bye',
    'hmm'
]
embedding=OpenAIEmbeddings(model='test-embedding-3-large',dimensions=32)
'''result=embedding.embed_query('Delhi is the capital of india')'''
result=embedding.embed_documents(docs)
print(str(result))