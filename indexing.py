from langchain.document_loaders import DirectoryLoader
 
directory = '/content/data'
 
def load_docs(directory):
 loader = DirectoryLoader(directory)
 documents = loader.load()
 return documents
 
documents = load_docs(directory)
len(documents)
from langchain.text_splitter import RecursiveCharacterTextSplitter
 
def split_docs(documents,chunk_size=500,chunk_overlap=20):
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
 docs = text_splitter.split_documents(documents)
 return docs
 
docs = split_docs(documents)
print(len(docs))
from langchain.embeddings import SentenceTransformerEmbeddings
 
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain.pinecone import PineconeIndexer
 
def index_embeddings(embeddings, docs):
    indexer = PineconeIndexer(api_key='votre-clé-api-pinecone', index_name='votre-nom-d-index')
    indexer.index(embeddings, docs)
 
index_embeddings(embeddings, docs)
