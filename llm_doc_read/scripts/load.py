## script for QnA advisor of the reseach papers

import streamlit as streamlit
import torch
#from langchain.prompts import PromptTemplate # ask for prompts
#from langchain.llms import CTransformers

#from langchain_community.llms import CTransfromers

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

#function to call local LLAM2-7b-chat model

#def getLlamaresp(inpput_txt,)


#load the file pdf file
def read_pdf(file):
    loader = PyPDFLoader(file)
    raw_doc = loader.load()
    print("len of docs",len(raw_doc))
    
    #chunk the file 
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
    )

    doc = text_splitter.split_documents(raw_doc)
    print ("length of chunked doc" , len(doc))

    ## get an embedding model from higging face to tokenise tbe push the data in local
    # vectorstore chromadb
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
    )
    #db = Chroma.from_documents(doc, embeddings, persist_directory="db")

    ##def get_vector_store(text_chunks):
    ##embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    vector_store = Chroma.from_documents(doc,embeddings, persist_directory="./chroma_db")

    #vector_store = FAISS.from_documents(doc, embedding=embeddings)
    #vector_store.save_local("faiss_index")
    query = "What is attention"
    docs = vector_store.similarity_search(query)
    print(docs[0].page_content)

def query_reader(query,db_dir):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
    )
    db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    docs = db.similarity_search(query)
    print(docs[0].page_content)


if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    #read_pdf("data/transformers_arxiv.pdf")
    db_dir="./chroma_db"
    query = "what is encoder decoder"

    query_reader(query,db_dir)

## from langchain_community.document_loaders import OnlinePDFLoader
##loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")

