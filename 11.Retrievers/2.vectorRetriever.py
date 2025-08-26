from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# 1. Prepare documents
docs = [
    Document(page_content="India and Pakistan have a complex geopolitical history.", metadata={"field":"geopolitics"}),
    Document(page_content="China's perspective on South Asia focuses on trade and security.", metadata={"field":"geopolitics"}),
    Document(page_content="The India-Pakistan conflicts have influenced regional policies.", metadata={"field":"geopolitics"}),
    Document(page_content="China invests heavily in infrastructure projects in Pakistan.", metadata={"field":"economy"})
]

# 2. Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create Chroma vector store
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_db"  # persist embeddings
)

# 4. Get retriever from vector store
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# 5. Query via retriever
query = "Geopolitical relationship between India, Pakistan, and China"
results = retriever.get_relevant_documents(query)

# 6. Print results
for i, doc in enumerate(results):
    print(f"Document {i+1}:\n{doc.page_content}\nMetadata: {doc.metadata}\n{'-'*50}")
