from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import OllamaEmbeddings  

import os

current_folder = os.getcwd()

DATA_PATH = os.path.join(current_folder, "data")

DB_PATH = "vectorstores/db/"

def create_vector_db():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Processed {len(documents)} lines in pdf file")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
    texts=text_splitter.split_documents(documents)
    vectorize = Chroma.from_documents(documents=texts, embedding=GPT4AllEmbeddings(),persist_directory=DB_PATH)      
    vectorize.persist()
    vectorize = None

if __name__=="__main__":
    create_vector_db()