# create_index.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Step 1: Load PDF
pdf_path = "medical.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Step 3: Generate embeddings
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

# Step 4: Create FAISS vector store and save it
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local("faiss_index")

print("✅ FAISS index saved to 'faiss_index/' folder.")
