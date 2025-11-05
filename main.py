
### 🧩 main.py
```python
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torch
import requests
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

app = FastAPI(title="AI Media Similarity Detector")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class ImagePair(BaseModel):
    image1: str
    image2: str

@app.post("/compare")
def compare_images(data: ImagePair):
    image1 = Image.open(BytesIO(requests.get(data.image1).content))
    image2 = Image.open(BytesIO(requests.get(data.image2).content))
    inputs = processor(images=[image1, image2], return_tensors="pt")
    features = model.get_image_features(**inputs)
    sim = torch.nn.functional.cosine_similarity(features[0], features[1], dim=0).item()
    return {"similarity_score": round(sim, 3)}
  
