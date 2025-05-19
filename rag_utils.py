import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

INDEX_DIR = "faiss_index"
PDF_PATH = "data/paper.pdf"

def load_or_create_faiss_index():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(INDEX_DIR):
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(INDEX_DIR)

    return db.as_retriever()

def get_rag_chain():
    retriever = load_or_create_faiss_index()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return rag_chain
