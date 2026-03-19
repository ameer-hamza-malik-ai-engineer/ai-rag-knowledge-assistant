import os
from pydoc import doc
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

print("Starting ingestion pipeline...")

def load_documents(docs_path = "docs"):

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory '{docs_path}' does not exist.")


    loader = DirectoryLoader(path = docs_path, glob = "*.txt", show_progress=True, silent_errors=True, loader_cls=TextLoader)

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(f"No documents found in directory '{docs_path}' with glob pattern '*.txt'.")

    for i, doc in enumerate(documents):
        print(f"Document {i}: {doc.metadata['source']} (length: {len(doc.page_content)} characters)")
        print(f"Content preview: {doc.page_content[:100]}...")  # Print the first 200 characters of the content
    
    return documents


def main():
    documents = load_documents()

if __name__ == "__main__":
    main()