from langchain_community.document_loaders import PyMuPDFLoader


def load_data(path="data/Medical_book.pdf"):
    loader = PyMuPDFLoader(file_path=path)
    data=loader.load()
    return data