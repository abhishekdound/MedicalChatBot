import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings


class Embedder:
    def __init__(self):
        self.embeddings = FastEmbedEmbeddings()
        self.folder_path = "vectorstore"
        self.index_name = "medical_book_index"

    def get_retriever(self):

        
        vectorstore = FAISS.load_local(
            folder_path=self.folder_path,
            embeddings=self.embeddings,
            index_name=self.index_name,
            allow_dangerous_deserialization=True  # required in new versions
        )


        return vectorstore.as_retriever(search_kwargs={"k":3})
