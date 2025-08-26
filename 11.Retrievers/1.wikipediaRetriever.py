from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(top_k_results=2,lang='en')
query='Geopolitical history of india and pakistan from the perspective of a chinese'
docs=retriever.invoke(query)
for i,doc in enumerate(docs):
    print(f"Document {i+1}:\n{doc.page_content}\n{'-'*50}")