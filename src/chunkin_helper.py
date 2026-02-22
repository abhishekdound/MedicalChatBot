from langchain_text_splitters import RecursiveCharacterTextSplitter
from .data_load_helper import load_data

def chunk_document():
    data=load_data()
    chunks_splitter=RecursiveCharacterTextSplitter(separators=['\n\n','\n','.'],chunk_size=1000,chunk_overlap=150)
    chunks=chunks_splitter.split_documents(documents=data)
    return chunks
