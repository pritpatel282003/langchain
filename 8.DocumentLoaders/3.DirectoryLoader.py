from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    path="D:/GEN_AI/langchain/8.DocumentLoaders/test_pdf",
    glob="*.txt",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8")
)

'''docs = loader.load()
for doc in docs:
    print(doc.metadata)'''

docs = loader.lazy_load()
for doc in docs:
    print(doc.metadata)

