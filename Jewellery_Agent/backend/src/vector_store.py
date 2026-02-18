from qdrant_client import QdrantClient
from colpali_engine.models import ColPali, ColPaliProcessor
from pdf2image import convert_from_path

import torch
import os


class VisualRetriever:
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        self.collection_name = "guide_documents"

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.colpali_model = ColPali.from_pretrained(
            "vidore/colpali-v1.2",
            torch_dtype=dtype,
            device_map=self.device
        ).eval()

        self.processor = ColPaliProcessor.from_pretrained(
            "vidore/colpali-v1.2")

    def retrieve_context_pages(self, query_text, k):
        with torch.no_grad():
            batch_query = self.processor.process_queries(
                [query_text]).to(self.colpali_model.device)
            query_embedding = self.colpali_model(**batch_query)

        multivector_query = query_embedding[0].cpu().float().numpy().tolist()

        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=multivector_query,
            using="colpali",
            limit=k
        ).points

        context_images = []

        for point in search_result:
            pdf_source = point.payload.get("source")
            page_num = point.payload.get("page_num")

            pdf_path = os.path.join("documents", f"{pdf_source}.pdf")

            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1
            )

            if images:
                context_images.append(images[0])

            print(f"retrieved page {page_num + 1} from {pdf_source}")

        return context_images
