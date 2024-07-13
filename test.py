import os
import io
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import faiss
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import torch.nn.functional as F
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

app = FastAPI()
dimension = 37  # Make sure this matches your model's output dimension
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIDMap(quantizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device).eval()

os.makedirs('uploaded_images', exist_ok=True)

def generate_16_digit_int():
    return random.randint(10**15, 10**16 - 1)

def extract_features(img):
    transform = transforms.Compose([
        transforms.Grayscale(),              # Convert image to grayscale
        transforms.Resize((28, 28)),         # Resize to match input size of model
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().numpy().flatten(), features.argmax(dim=1).item()

def setup_faiss_index():
    for filename in os.listdir('uploaded_images'):
        image_path = os.path.join('uploaded_images', filename)
        img = Image.open(image_path)
        vector, _ = extract_features(img)
        image_id = int(filename.split('.')[0])
        index.add_with_ids(np.array([vector], dtype='float32'), np.array([image_id]))

@app.on_event("startup")
async def startup_event():
    setup_faiss_index()
    print("Faiss index has been populated with existing images.")

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    vector, pred = extract_features(img)
    image_id = generate_16_digit_int()
    image_path = f'uploaded_images/{image_id}.png'
    img.save(image_path)
    index.add_with_ids(np.array([vector], dtype='float32'), np.array([image_id]))
    return JSONResponse(content={"message": "Image uploaded and features extracted.", "ID": str(image_id), "prediction": pred - 1})

@app.post("/search/")
async def search_image(file: UploadFile = File(...)):
    content = await file.read()
    img = Image.open(io.BytesIO(content))
    query_vector, _ = extract_features(img)
    D, I = index.search(np.array([query_vector], dtype='float32'), k=5)
    for i in range(5):
        if D.size > 0 and D[0][i] > 0:
            image_id = I[0][i]
            image_path = f'uploaded_images/{image_id}.png'
            if os.path.exists(image_path):
                return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="No similar image found.")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
