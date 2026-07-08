from fastapi import FastAPI
from fastembed import ImageEmbedding
from pydantic import BaseModel
import requests
from PIL import Image
from io import BytesIO

app = FastAPI()
# This uses the small CLIP ViT-B/32 model by default
model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")

class ImageUrl(BaseModel):
	url: str

@app.post("/embed")
def get_embedding(data: ImageUrl):
	response = requests.get(data.url)
	image = Image.open(BytesIO(response.content))

	# Generate embeddings
	embeddings = list(model.embed([image]))
	return embeddings[0].tolist()
