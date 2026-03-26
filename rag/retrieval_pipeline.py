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

client = InferenceClient(
    provider="fireworks-ai",
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

print(completion.choices[0].message.content)
