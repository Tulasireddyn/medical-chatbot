import os
from dotenv import load_dotenv
from utils.pdf_loader import load_and_split_pdf
from utils.embedding import get_embeddings
from utils.pinecone_utils import init_pinecone

load_dotenv()
init_pinecone()

pdf_path = r"C:\Users\tulas\OneDrive\Desktop\project\MEDICAL CHATBOT\data\medical_book.pdf"
chunks = load_and_split_pdf(pdf_path)
embed_fn = get_embeddings()

# Create texts and metadata from chunks
texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]

# Generate embeddings manually
vectors = embed_fn(texts)

# Push to Pinecone
import pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pinecone.Index(os.getenv("INDEX_NAME"))

for i, (vec, meta) in enumerate(zip(vectors, metadatas)):
    index.upsert([
        (f"doc-{i}", vec, meta)
    ])
