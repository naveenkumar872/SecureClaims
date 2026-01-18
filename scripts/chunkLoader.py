from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import torch
load_dotenv()
import os

QUADRANT_URL = os.getenv("QUADRANT_URL")
QUADRANT_API_KEY = os.getenv("QUADRANT_API_KEY")

qdrant_client = QdrantClient(
    url=QUADRANT_URL, 
    api_key=QUADRANT_API_KEY,
)

# Setup embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
print(f"Using device: {device}")

COLLECTION_NAME = "insurance_policy_chunks"
EMBEDDING_DIM = 384  # bge-small-en-v1.5 dimension

# Read chunks from Final.txt
with open("Final.txt", "r", encoding="utf-8") as f:
    content = f.read()
    chunks = content.split("\n\n===\n\n")

print(f"Loaded {len(chunks)} chunks")

# Create collection if it doesn't exist
collections = qdrant_client.get_collections().collections
collection_names = [c.name for c in collections]

if COLLECTION_NAME not in collection_names:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection: {COLLECTION_NAME}")
else:
    print(f"Collection {COLLECTION_NAME} already exists")

# Generate embeddings and upload to Qdrant
points = []
for i, chunk in enumerate(chunks):
    if chunk.strip():  # Skip empty chunks
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        embedding = embed_model.encode(chunk, normalize_embeddings=True).tolist()
        
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={"text": chunk, "chunk_index": i}
        ))

# Upload points in batches
BATCH_SIZE = 100
for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i+BATCH_SIZE]
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch
    )
    print(f"Uploaded batch {i//BATCH_SIZE + 1}/{(len(points) + BATCH_SIZE - 1)//BATCH_SIZE}")

print(f"Successfully uploaded {len(points)} chunks to Qdrant!")
print(qdrant_client.get_collection(COLLECTION_NAME))