from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(collection_name="langchain", embedding_function=embedding_model, persist_directory=persistent_directory,
collection_metadata={"hnsw:space": "cosine"})

query = "How much Microsoft pay to acquire Github?"

retriever = db.as_retriever(search_kwargs={"k": 3})

# retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5,"score_threshold": 0.3})

relevant_docs = retriever.invoke(query)

print("Query:", query)

print("--Context--")

for i, doc in enumerate(relevant_docs):
    print(f"Document {i}: {doc.metadata['source']} (length: {len(doc.page_content)} characters)")
print("Relevant Documents: ", relevant_docs)