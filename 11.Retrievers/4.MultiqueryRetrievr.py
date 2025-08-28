import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI 

# 1. Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY not set. Add it to your environment or .env file.")

# 2. Documents (health + 2 black hole docs)
docs = [
    Document(page_content="Regular exercise helps reduce the risk of heart disease.", metadata={"topic": "fitness"}),
    Document(page_content="A balanced diet with fruits and vegetables strengthens immunity.", metadata={"topic": "nutrition"}),
    Document(page_content="Diabetes management requires monitoring blood sugar levels.", metadata={"topic": "disease"}),
    Document(page_content="Mental health can be improved by mindfulness and meditation.", metadata={"topic": "mental_health"}),
    Document(page_content="Vaccinations are essential to prevent infectious diseases.", metadata={"topic": "preventive_health"}),
    Document(page_content="High blood pressure can be managed with lifestyle changes and medication.", metadata={"topic": "disease"}),
    Document(page_content="Sleep plays a crucial role in memory and overall well-being.", metadata={"topic": "wellness"}),
    Document(page_content="Obesity increases the risk of diabetes, hypertension, and heart disease.", metadata={"topic": "nutrition"}),
    Document(page_content="Hydration is vital for kidney function and maintaining body temperature.", metadata={"topic": "wellness"}),
    Document(page_content="Smoking is a major risk factor for lung cancer and heart problems.", metadata={"topic": "preventive_health"}),
    Document(page_content="A black hole is a region of space where gravity is so strong that not even light can escape.", metadata={"topic": "blackhole"}),
    Document(page_content="The first direct image of a black hole was captured in 2019 by the Event Horizon Telescope.", metadata={"topic": "blackhole_observation"}),
]

# 3. Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Persistent Chroma DB
persist_dir = "./chroma_health_db"
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

# 5. Base retriever with MMR
base_retriever = vector_store.as_retriever(
    search_type="mmr",           # ✅ use MMR
    search_kwargs={"k": 6, "fetch_k": 12}  # fetch_k > k recommended for MMR
)

# 6. Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=512,
    google_api_key=GOOGLE_API_KEY
)

# 7. MultiQueryRetriever
multiquery_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

# 8. Query
query = "How can lifestyle changes improve overall health and reduce disease risk?"
results = multiquery_retriever.invoke(query)

# 9. Print results
for i, doc in enumerate(results, 1):
    print(f"📄 Result {i}: {doc.page_content} | Metadata: {doc.metadata}")
