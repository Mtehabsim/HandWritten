import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import io
import faiss

app = FastAPI()

# Initialize Faiss index (assuming 10-dimension embeddings)
dimension = 64
index = faiss.IndexFlatL2(dimension)

# Image directory for storing uploads
image_directory = "uploaded_images"
os.makedirs(image_directory, exist_ok=True)

# Dictionary to map image IDs to their file paths
image_id_to_path = {}

# Load your pre-trained model
model = load_model('features.h5', compile=False)

def extract_features(img, model):
    """Extract features from an image using the loaded CNN model."""
    img = img.convert('L').resize((28, 28))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    features = model.predict(img_array)
    return features.flatten()

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Save uploaded image, process it, and save its features."""
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    identify_model = load_model('mnist_cnn_model.h5', compile=False)
    vector = extract_features(img, identify_model)
    vector = vector.tolist()
    value = str(vector.index(max(vector)))
                
    vector = extract_features(img, model)

    # Save the image with a unique ID
    image_id = str(uuid.uuid4())
    image_path = os.path.join(image_directory, f"{image_id}.png")
    img.save(image_path)

    # Add image path to the dictionary
    image_id_to_path[image_id] = image_path

    # Add vector to Faiss index
    index.add(np.array([vector], dtype='float32'))
    
    return JSONResponse(content={"message": "Image uploaded successfully", "image_id": image_id, "value": value})

@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    query_vector = extract_features(img, model)

    # Query the Faiss index for the most similar images
    D, I = index.search(np.array([query_vector], dtype='float32'), k=1)
    print(f"Distances: {D.tolist()}")
    print(f"Indices: {I.tolist()}")

    nearest_image_id = list(image_id_to_path.keys())[I[0][0]]
    nearest_image_path = image_id_to_path[nearest_image_id]

    return FileResponse(nearest_image_path)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
