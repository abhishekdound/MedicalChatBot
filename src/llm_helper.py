import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .embedding_helper import Embedder

load_dotenv()
llm=ChatGroq(model=os.getenv("GROQ_MODEL")
             ,temperature=0.1)

retriever=Embedder().get_retriever()

prompt = ChatPromptTemplate.from_template("""
You are a helpful medical assistant.

Answer ONLY from the provided context.
If answer not found, say "I don't know".

Context:
{context}

Question:
{question}
""")

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


rag_chain = (
    {
        "context": retriever | format_docs,   # retrieve → format
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is cancer?")
print(response)
