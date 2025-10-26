import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import io
import time
import os
from collections import OrderedDict
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CROP_MODEL = {
    "name": "Crop MobileNetV3-S",
    "type": "crop",
    "model_func": models.mobilenet_v3_small,
    "state_dict_path": os.path.join(BASE_DIR, "model", "crop", "mobilenetv3small_best_crop_8020.pth"),
    "class_names_file": os.path.join(BASE_DIR, "model", "crop", "class_names_crop.txt"),
    "classifier_idx": 3,
}
PRODUCE_MODEL = {
    "name": "Produce EfficientNetV2-S",
    "type": "produce",
    "model_func": models.efficientnet_v2_s,
    "state_dict_path": os.path.join(BASE_DIR, "model", "produce", "efficientnetv2s_best_produce.pth"),
    "class_names_file": os.path.join(BASE_DIR, "model", "produce", "class_names_produce.txt"),
    "classifier_idx": 1,
}

# --- DEVICE SETUP ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- MODEL CACHE ---
MODEL_CACHE = {}

# --- HELPER FUNCTIONS ---

def load_model(model_config):
    if model_config["name"] in MODEL_CACHE:
        return MODEL_CACHE[model_config["name"]]

    print(f"Loading model: {model_config['name']}...")
    
    model = model_config["model_func"](weights=None)
    
    # 2. Get number of classes from the label file
    with open(model_config["class_names_file"], 'r') as f:
        class_names = [line.strip() for line in f]
    num_classes = len(class_names)
    
    try:
        classifier_layer = model.classifier[model_config["classifier_idx"]]
        in_features = classifier_layer.in_features
        model.classifier[model_config["classifier_idx"]] = torch.nn.Linear(in_features, num_classes)
    except Exception as e:
        print(f"ERROR: Could not modify model architecture for {model_config['name']}: {e}")
        raise HTTPException(status_code=500, detail=f"Model setup failed for {model_config['name']}")

    # 4. Load the trained state dictionary
    try:
        state_dict = torch.load(
            model_config["state_dict_path"],
            map_location=DEVICE
        )
        
        # Correct the key names if they include 'module.'
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)

    except Exception as e:
        print(f"ERROR: Could not load state dictionary for {model_config['name']}: {e}")
        raise HTTPException(status_code=500, detail=f"Corrupted model file: {model_config['state_dict_path']}")

    # 5. Final setup and caching
    model.to(DEVICE)
    model.eval()
    
    # Store in cache
    MODEL_CACHE[model_config["name"]] = {
        "model": model,
        "class_names": class_names,
        "transform": get_transform()
    }
    
    print(f"Model {model_config['name']} loaded successfully.")
    return MODEL_CACHE[model_config["name"]]

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def predict_image(image_bytes, model_info):
    model = model_info["model"]
    class_names = model_info["class_names"]
    transform = model_info["transform"]
    
    # Load image from bytes
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Preprocess
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(DEVICE)

    # Inference
    with torch.no_grad():
        start_time = time.perf_counter()
        output = model(input_batch)
        end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000
    
    # Post-process
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    conf, predicted_class_idx = torch.max(probabilities, 0)
    
    # Create the full probabilities dictionary
    probabilities_list = probabilities.cpu().tolist()
    full_predictions_dict = dict(zip(class_names, probabilities_list))
    
    # Return all necessary data points
    return {
        "top_prediction": class_names[predicted_class_idx.item()],
        "confidence_percent": round(conf.item() * 100, 2),
        "avg_latency_ms": round(latency_ms, 2), 
        "full_predictions": full_predictions_dict
    }

# --- FASTAPI SETUP ---
app = FastAPI(title="ML Inference API")

# STEP 1: ADD CORS MIDDLEWARE
origins = [
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# STEP 2: Use the startup event to load models ONCE
@app.on_event("startup")
def startup_event():
    print("Server startup: Loading ML Models...")
    try:
        load_model(CROP_MODEL)
        load_model(PRODUCE_MODEL)
        print("✅ All models pre-loaded.")
    except HTTPException as e:
        print(f"❌ CRITICAL ERROR: Failed to pre-load a model: {e.detail}. Server will run with partial functionality.")

@app.get("/ping")
def ping():
    return {"status": "ok", "device": str(DEVICE), "models_loaded": list(MODEL_CACHE.keys())}


# --- API ENDPOINT ---
@app.post("/predict")
async def get_produce(
    file: UploadFile = File(..., description="The image file to analyze."),
    status: str = Form(..., description="Classification type: 'produce' or 'crop'.")
):
    """
    Receives an image and a status ('produce' or 'crop') and returns a prediction 
    in the requested format.
    """
    
    # 1. Input Validation and Model Selection
    status = status.lower()
    if status == "produce":
        model_config = PRODUCE_MODEL
    elif status == "crop":
        model_config = CROP_MODEL
    else:
        raise HTTPException(
            status_code=400, 
            detail="Invalid status value. Must be 'produce' or 'crop'."
        )

    # 2. Load the image file bytes
    image_bytes = await file.read()
    
    # 3. Load/Get the correct model
    try:
        model_info = load_model(model_config)
    except HTTPException as e:
        raise e 

    # 4. Run Prediction
    try:
        prediction_result = predict_image(image_bytes, model_info)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # 5. Return result in the requested JSON format
    return JSONResponse(content={
        "model_name": model_config["name"],
        "model_type": model_config["type"],
        "test_image": file.filename, 
        
        # Data from predict_image function
        "top_prediction": prediction_result["top_prediction"],
        "confidence_percent": prediction_result["confidence_percent"],
        "avg_latency_ms": prediction_result["avg_latency_ms"],
        "full_predictions": prediction_result["full_predictions"]
    })