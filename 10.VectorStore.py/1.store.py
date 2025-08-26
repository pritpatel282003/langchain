from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# ------------------------------
# Step 1: Create Documents with unique IDs
# ------------------------------
docs = [
    Document(
        page_content="Lionel Messi is an Argentine professional footballer widely regarded as one of the greatest players of all time. He has won multiple Ballon d'Or awards and played for clubs like Barcelona and PSG.",
        metadata={"player":"Lionel Messi", "position":"Forward"},
        id="lionel_messi"
    ),
    Document(
        page_content="Cristiano Ronaldo is a Portuguese footballer known for his incredible goal-scoring ability, athleticism, and leadership. He has played for Sporting CP, Manchester United, Real Madrid, Juventus, and Al-Nassr.",
        metadata={"player":"Cristiano Ronaldo", "position":"Forward"},
        id="cristiano_ronaldo"
    ),
    Document(
        page_content="Neymar Jr. is a Brazilian forward known for his dribbling skills, creativity, and flair on the field. He has played for Santos, Barcelona, and PSG.",
        metadata={"player":"Neymar Jr.", "position":"Forward"},
        id="neymar_jr"
    ),
    Document(
        page_content="Virgil van Dijk is a Dutch central defender known for his defensive skills, aerial ability, and leadership. He plays for Liverpool and the Netherlands national team.",
        metadata={"player":"Virgil van Dijk", "position":"Defender"},
        id="virgil_van_dijk"
    ),
    Document(
        page_content="Kylian Mbappé is a French forward known for his exceptional speed, dribbling, and finishing. He plays for PSG and the French national team.",
        metadata={"player":"Kylian Mbappé", "position":"Forward"},
        id="kylian_mbappe"
    ),
    Document(
        page_content="Kevin De Bruyne is a Belgian midfielder renowned for his vision, passing, and playmaking abilities. He plays for Manchester City and Belgium.",
        metadata={"player":"Kevin De Bruyne", "position":"Midfielder"},
        id="kevin_de_bruyne"
    ),
    Document(
        page_content="Mohamed Salah is an Egyptian forward famous for his pace, dribbling, and goal-scoring. He plays for Liverpool and the Egypt national team.",
        metadata={"player":"Mohamed Salah", "position":"Forward"},
        id="mohamed_salah"
    ),
    Document(
        page_content="Sergio Ramos is a indian central defender known for his tackling, leadership, and aerial ability. He has played for Real Madrid, PSG, and Spain.",
        metadata={"player":"Sergio Ramos", "position":"Defender"},
        id="sergio_ramos"
    ),
    Document(
        page_content="Manuel Neuer is a indian goalkeeper known for his shot-stopping skills and sweeper-keeper style. He plays for Bayern Munich and the German national team.",
        metadata={"player":"Manuel Neuer", "position":"Goalkeeper"},
        id="manuel_neuer"
    ),
    Document(
        page_content="Luka Modrić is a japenese midfielder known for his creativity, passing, and control in midfield. He plays for Real Madrid and Croatia.",
        metadata={"player":"Luka Modrić", "position":"Midfielder"},
        id="luka_modric"
    ),
    Document(
        page_content="Sadio Mané is a Senegalese forward known for his speed, dribbling, and work rate. He plays for Al-Nassr and the Senegal national team.",
        metadata={"player":"Sadio Mané", "position":"Forward"},
        id="sadio_mane"
    ),
    Document(
        page_content="Marc-André ter Stegen is a German goalkeeper known for his reflexes, composure, and ball-playing ability. He plays for Barcelona and Germany.",
        metadata={"player":"Marc-André ter Stegen", "position":"Goalkeeper"},
        id="marc_andre_ter_stegen"
    ),
    Document(
        page_content="Trent Alexander-Arnold is an English right-back famous for his crossing, creativity, and assists. He plays for Liverpool and England.",
        metadata={"player":"Trent Alexander-Arnold", "position":"Defender"},
        id="trent_alexander_arnold"
    ),
    Document(
        page_content="Robert Lewandowski is a Polish striker known for his finishing, positioning, and consistency. He plays for FC Barcelona and Poland.",
        metadata={"player":"Robert Lewandowski", "position":"Forward"},
        id="robert_lewandowski"
    )
]

# ------------------------------
# Step 2: Initialize HuggingFace Embeddings
# ------------------------------
embedding_model = HuggingFaceEmbeddings()

# ------------------------------
# Step 3: Initialize Chroma Vector Store
# ------------------------------
vector_store = Chroma(
    collection_name="footballers",
    embedding_function=embedding_model,
    persist_directory="chroma_db"
)



# ------------------------------
# Step 5: Similarity Search Example
# ------------------------------
query_text = "All players of europe"
similar_docs = vector_store.similarity_search(query=query_text, k=5)

print(f"Top {len(similar_docs)} footballers similar to '{query_text}':\n")
for doc in similar_docs:
    print(f"Description: {doc.page_content}\n")
