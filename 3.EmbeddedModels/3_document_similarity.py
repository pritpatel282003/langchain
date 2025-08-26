from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Original embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
cricketers = [
    "Virat Kohli is a consistent run-scorer known for chasing targets.",
    "Rohit Sharma is famous for explosive double centuries in ODIs.",
    "Steve Smith is a technically strong batsman in Test cricket.",
    "Kane Williamson anchors the innings with calm and steady batting.",
    "Joe Root dominates in Test cricket with elegant stroke play.",
    "David Warner is an aggressive opener with quick scoring ability.",
    "Babar Azam is stylish and reliable across all formats of cricket.",
    "MS Dhoni is a finisher known for calm leadership under pressure.",
    "AB de Villiers plays 360-degree shots with innovative stroke play.",
    "Chris Gayle is a power-hitter capable of massive six-hitting sprees."
]

query='best in all cricket format'
docs_embedding = embeddings.embed_documents(cricketers)
query_embedding = embeddings.embed_query(query)

scores=cosine_similarity([query_embedding],docs_embedding)[0]
index,score=sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(cricketers[index])
print("score",score)



