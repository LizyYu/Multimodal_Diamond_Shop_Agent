import chromadb
import pandas as pd
import requests
import ast
import time

from sentence_transformers import SentenceTransformer
from PIL import Image
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Optional


class JewelleryMetaData(BaseModel):
    material: Optional[str] = Field(
        description="The primary metal material, e.g. rose gold, Platinum, Yellow Gold")
    style: Optional[str] = Field(
        description="The artistic style, e.g., Vintage, Modern, Solitaire")
    gemstone: Optional[str] = Field(
        description="The main stone, e.g., Diamond, Sapphire")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", api_key="")
structured_llm = llm.with_structured_output(JewelleryMetaData)

model = SentenceTransformer("clip-ViT-B-32")
client = chromadb.PersistentClient(path="./blue_nile_agentic_db")

product_collection = client.get_or_create_collection(name="product_knowledge")
visual_collection = client.get_or_create_collection(name="visual_index")

df = pd.read_csv("blue_nile.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)

for index, row in df.iterrows():
    product_id = str(index)

    raw_text = ""
    for name, value in row.items():
        if "url" not in name:
            raw_text += f"{name}: {value}. "

    extracted_data = structured_llm.invoke(raw_text)

    metadata_dict = {
        "product_id": product_id,
        "name": row["name"],
        "price": float(row["price"]),
        "material": extracted_data.material or "Unknown",
        "style": extracted_data.style or "Unknown",
        "gemstone": extracted_data.gemstone or "Unknown"
    }

    product_collection.add(
        ids=[product_id],
        documents=[raw_text],
        metadatas=[metadata_dict]
    )

    image_urls = ast.literal_eval(row["image_url"])

    for image_idx, image_url in enumerate(image_urls):
        response = requests.get(image_url, stream=True, timeout=10)
        image = Image.open(BytesIO(response.content))
        vector = model.encode(image).tolist()

        visual_collection.add(
            ids=[f"{product_id}_{image_idx}"],
            embeddings=[vector],
            metadatas=[{
                "parent_id": product_id,
                "name": row["name"],
                "price": row["price"],
                "url": row["url"],
                "image_url": image_url,
                "view_index": image_idx
            }]
        )

    print(f"Processed {index} items.")
print("Done.")
