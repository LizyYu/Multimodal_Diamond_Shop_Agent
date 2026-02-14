import torch
import os
import uuid

from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client.http import models


class ColPaliRAGDB:
    def __init__(self, collection_name):
        self.collection_name = collection_name

        self.client = QdrantClient(url="http://localhost:6333")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        self.processor = ColPaliProcessor.from_pretrained(
            "vidore/colpali-v1.2")

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "colpali": models.VectorParams(
                        size=128,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    )
                }
            )

    def ingest_pdf(self, pdf_path):
        print(f"Processing {pdf_path}...")
        images = convert_from_path(pdf_path)

        batch_size = 4
        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]

            with torch.no_grad():
                processed_images = self.processor.process_images(
                    batch).to(self.device)
                embeddings = self.model(**processed_images)

            points = []
            for j, img_embedding in enumerate(embeddings):
                multi_vectors = img_embedding.cpu().float().numpy().tolist()
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"colpali": multi_vectors},
                    payload={"page_num": i + j + 1,
                             "source": pdf_path.split("/")[-1][:-4]}
                ))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        print(f"Ingested {len(images)} pages.")


"""
client = QdrantClient(url="http://localhost:6333")
client.delete_collection("guide_documents")

db_builder = ColPaliRAGDB(collection_name="guide_documents")

for pdf_path in os.listdir("./documents"):
    pdf_path = os.path.join("./documents", pdf_path)
    db_builder.ingest_pdf(pdf_path)
"""


COLLECTION_NAME = "guide_documents"
MODEL_NAME = "vidore/colpali-v1.2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Connecting to Qdrant (Docker) ...")
client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists(COLLECTION_NAME):
    print(f"Error: collection '{COLLECTION_NAME}' not found in Qdrant!")
    import sys
    sys.exit()

print("Loading ColPali Model for query embedding ...")
model = ColPali.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE
)
processor = ColPaliProcessor.from_pretrained(MODEL_NAME)

query_text = "which diamond is very sparkling"
print(f"Query: '{query_text}'")

with torch.no_grad():
    batch_query = processor.process_queries([query_text]).to(DEVICE)
    query_embedding = model(**batch_query)

    multivector = query_embedding[0].cpu().float().numpy().tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=multivector,
    using="colpali",
    limit=3
).points

print("Top 3 Retrieve Results:")
for i, hit in enumerate(results):
    print(f"{i + 1}. Score: {hit.score:.4f} | Page: {hit.payload.get('page_num')} | Source: {hit.payload.get('source')}")
