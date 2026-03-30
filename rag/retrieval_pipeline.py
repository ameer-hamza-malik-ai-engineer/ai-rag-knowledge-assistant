import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from huggingface_hub import InferenceClient

load_dotenv()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def llm(query, relevant_docs):
    client = InferenceClient(
        provider="fireworks-ai",
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
    )

    completion = client.chat.completions.create(
        model="MiniMaxAI/MiniMax-M2.5",
        messages=[
            {
                "role": "user",
                "content": prompt.format(context=format_docs(relevant_docs), question=query)
            }
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return completion.choices[0].message.content

def query_with_history(query, chat_history):
    client = InferenceClient(
        provider="fireworks-ai",
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

    prompt = ChatPromptTemplate.from_template(
        "Given the chat history, rewrite the new question to be standalone and searchable. Just rewrite the question:\n\n{context}\n\nQuestion: {question}"
    )

    completion = client.chat.completions.create(
        model="MiniMaxAI/MiniMax-M2.5",
        messages=[
            {
                "role": "user",
                "content": prompt.format(context=chat_history, question=query)
            }
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    persistent_directory = "db/chroma_db"

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(collection_name="langchain", embedding_function=embedding_model, persist_directory=persistent_directory,
    collection_metadata={"hnsw:space": "cosine"})

    retriever = db.as_retriever(search_kwargs={"k": 5})
    chat_history = []
    answer = ""

    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        if chat_history == []:
            relevant_docs = retriever.invoke(query)
            answer = llm(query, relevant_docs)
        else:
            rewritten_query = query_with_history(query, chat_history)
            print("Rewritten Query:", rewritten_query)
            relevant_docs = retriever.invoke(rewritten_query)
            answer = llm(rewritten_query, relevant_docs)

        chat_history.append((query, answer))
        print("Answer:", answer)
