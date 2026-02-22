import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from .chunkin_helper import chunk_document


class Embedder:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.folder_path = "vectorstore"
        self.index_name = "medical_book_index"
        self.documents=chunk_document()

    def get_retriever(self):

        if os.path.exists(self.folder_path):
            vectorstore = FAISS.load_local(
                folder_path=self.folder_path,
                embeddings=self.embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=True  # required in new versions
            )

        else:
            vectorstore = FAISS.from_documents(
                documents=self.documents,
                embedding=self.embeddings
            )

            vectorstore.save_local(
                folder_path=self.folder_path,
                index_name=self.index_name
            )

        return vectorstore.as_retriever(search_kwargs={"k":3})