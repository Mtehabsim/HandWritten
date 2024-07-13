import os
import uuid
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import numpy as np
import io
import faiss
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
app = FastAPI()

# Initialize Faiss index (check the correct dimension)
dimension = 37  # Make sure this matches your model's output dimensions
index = faiss.IndexFlatL2(dimension)

# Image directory for storing uploads
image_directory = "uploaded_images"
os.makedirs(image_directory, exist_ok=True)

# Dictionary to map image IDs to their file paths
image_id_to_path = {}

# Define a transform to convert the image data to a suitable format
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 37)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

def extract_features(img):
    """Extract features from an image using the loaded CNN model."""
    img = img.convert('L')  # Convert to grayscale
    img = transform(img).unsqueeze(0)  # Apply the transform and add batch dimension
    with torch.no_grad():
        features = model(img)
    return features.squeeze().numpy()  # Convert to numpy array for faiss

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Save uploaded image, process it, and save its features."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    vector = extract_features(img)

    # Save the image with a unique ID
    image_id = str(uuid.uuid4())
    image_path = os.path.join(image_directory, f"{image_id}.png")
    img.save(image_path)
    img.close()

    # Add image path to the dictionary
    image_id_to_path[image_id] = image_path
    
    # Add vector to Faiss index
    index.add(np.array([vector], dtype='float32'))
    vector = vector.tolist()
    target_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z']
    
    return JSONResponse(content={"message": "Image uploaded successfully", "image_id": image_id, "value": target_letters[vector.index(max(vector))]})

@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    """Find and return the most similar images to the uploaded one."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    query_vector = extract_features(img)

    # Query the Faiss index for the most similar images
    D, I = index.search(np.array([query_vector], dtype='float32'), k=1)
    if I.size == 0 or D[0][0] == float('inf'):
        return JSONResponse(content={"message": "No similar images found"})
    nearest_image_id = list(image_id_to_path.keys())[I[0][0]]
    nearest_image_path = image_id_to_path[nearest_image_id]

    return FileResponse(nearest_image_path)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
