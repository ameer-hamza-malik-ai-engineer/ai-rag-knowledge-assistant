import os
from pydoc import doc
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("Starting ingestion pipeline...")

def load_documents(docs_path = "docs"):

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")


    loader = DirectoryLoader(path = docs_path, glob = "*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in directory '{docs_path}' with glob pattern '*.txt'.")

    # for i, doc in enumerate(documents):
    #     print(f"Document {i}: {doc.metadata['source']} (length: {len(doc.page_content)} characters)")
    #     print(f"Content preview: {doc.page_content[:100]}...")  # Print the first 100 characters of the content
    
    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)

    # for i, doc in enumerate(split_docs):
    #     print(f"Chunk {i}: {doc.metadata['source']} (length: {len(doc.page_content)} characters)")
    #     print(f"Content preview: {doc.page_content[:100]}...")  # Print the first 100 characters of the content

    return split_docs

def create_vector_store(split_docs, collection_name="db/chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=collection_name, collection_metadata={"hnsw:space": "cosine"})
    return vector_store


def main():
    documents = load_documents()
    split_docs = split_documents(documents)
    vector_store = create_vector_store(split_docs)


if __name__ == "__main__":
    main()