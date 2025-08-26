from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text='Hi my name is prit !'
vector=embeddings.embed_query(text)
print(vector)