from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
import os

"""
Qdrant is not running as a server, only one instance can access it,
make sure to stop llm_web.py before you run this script
"""

collection_name = "documents"
embedding_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
cache_path = "/root/qdrant_cache"
base_dir = '/root/llm/files' # RAG Documents
documents = []

for file in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file)
    if file.endswith('.pdf'):
        documents.extend(PyPDFLoader(file_path).load())
    elif file.endswith('.docx'):
        documents.extend(Docx2txtLoader(file_path).load())
    elif file.endswith('.txt'):
        documents.extend(TextLoader(file_path).load())
print("Documents loaded")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)
print("Documents splited")

embeddings = HuggingFaceEmbeddings(model_name=embedding_name, show_progress=True)
print("Embeddings loaded")

vectorstore = Qdrant.from_existing_collection(
    embedding=embeddings,
    path=cache_path,
    collection_name=collection_name)

vectorstore.add_documents(chunked_documents)
print("Documents added")
