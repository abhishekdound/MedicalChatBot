import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .embedding_helper import Embedder

from .web_help import Web_search






class LLM:
    def __init__(self):
        self.prompt = ChatPromptTemplate.from_template("""
                                                        You are a helpful medical assistant.
                                                        
                                                        Answer ONLY from the provided context.
                                                        If answer not found, say "I don't know".
                                                        
                                                        Context:
                                                        {context}
                                                        
                                                        Question:
                                                        {question}
                                                        """)

        self.retriever = Embedder().get_retriever()
        load_dotenv()
        self.llm=ChatGroq(model=os.getenv("GROQ_MODEL")
             ,temperature=0.1)

        self.rag_chain = (
                {
                    "context": self.retriever | self.format_docs,
                    "question": RunnablePassthrough()
                }
                | self.prompt
                | self.llm
                | StrOutputParser()
        )



    def format_docs(self,docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", "")
            formatted.append(f"[Source: {source}, Page: {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def rag_result(self,question):




        response = self.rag_chain.invoke(question)

        if 'i don\'t know' in response.lower():
            web_prompt=Web_search().web_search(question)
            response=self.llm.invoke(web_prompt).content
        return response
