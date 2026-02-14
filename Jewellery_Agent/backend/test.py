import torch
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

model_name = "vidore/colpali-v1.2"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {model_name} ...")

model = ColPali.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map=device
).eval()
processor = ColPaliProcessor.from_pretrained(model_name)

dummy_image = Image.new("RGB", (224, 224), color="white")
query = "A plain white image"

with torch.no_grad():
    batch_images = processor.process_images([dummy_image]).to(device)
    batch_queries = processor.process_queries([query]).to(device)

    image_emb = model(**batch_images)
    text_emb = model(**batch_queries)

    score = processor.score(text_emb, image_emb)[0].item()

print(f"DIAGNOSE SCORE: {score:.4f}")
