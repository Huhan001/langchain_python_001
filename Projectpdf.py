import os 
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


def load_document(file):
    _, extension = os.path.splitext(file)
    if extension == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        print("Loading document {}".format(file))
        loader = PyPDFLoader(file)
    elif extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        print("Loading document {}".format(file))
        loader = Docx2txtLoader(file)
    elif extension == ".txt":
        from langchain_community.document_loaders import TxtLoader
        print("Loading document {}".format(file))
        loader = TxtLoader(file)
    elif extension == ".csv":
        from langchain_community.document_loaders import CSVLoader
        print("Loading document {}".format(file))
        loader = CSVLoader(file)

    data = loader.load()

    # chunking
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0, 
                                              separators=['. ', '\n', '\r\n', '!', '?', ';', ':','/', '(', ')', '[', ']',',','\t'],
                                              is_separator_regex=True)
    chunks = splitter.split_documents(data)
    return chunks


