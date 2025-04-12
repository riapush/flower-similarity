from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import torchvision

app = FastAPI(title="Flower Similarity Search API")

# Загрузка предварительно вычисленных признаков и путей к изображениям
with open("test_features.pkl", "rb") as f:
    test_features = pickle.load(f)

# Загрузка модели (ResNet18 без последнего слоя)
model = torchvision.models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Трансформации для входного изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(img_tensor):
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()

@app.post("/search")
async def search_similar_images(file: UploadFile = File(...)):
    try:
        # Загрузка и преобразование изображения
        img = Image.open(file.file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)
        
        # Извлечение признаков
        query_features = extract_features(img_tensor)
        
        # Поиск похожих изображений
        similarities = {}
        for img_path, features in test_features.items():
            sim = cosine_similarity(query_features.reshape(1, -1), features.reshape(1, -1))[0][0]
            similarities[img_path] = float(sim)
        
        # Сортировка и выбор топ-5
        top_5 = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return JSONResponse(content={"similar_images": top_5})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))