from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# 1. Documents
docs = [
    Document(page_content="India and Pakistan have a complex geopolitical history.", metadata={"field": "geopolitics"}),
    Document(page_content="China's perspective on South Asia focuses on trade and security.", metadata={"field": "geopolitics"}),
    Document(page_content="The India-Pakistan conflicts have influenced regional policies.", metadata={"field": "geopolitics"}),
    Document(page_content="China invests heavily in infrastructure projects in Pakistan.", metadata={"field": "economy"})
]

# 2. Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Create Chroma and store persistently
persist_dir = "./chroma_db"
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

print("✅ Vector DB stored in:", persist_dir)

# 4. Load existing Chroma DB
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

# 5. Use MMR retriever (diverse + relevant results)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "fetch_k": 5, "lambda_mult": 0.5}
)

query = "Geopolitical relationship between India, Pakistan, and China"
results = retriever.get_relevant_documents(query)

# 6. Print results
for doc in results:
    print("📄", doc.page_content, "| Metadata:", doc.metadata)
