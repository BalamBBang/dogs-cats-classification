import io
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from PIL import Image

app = FastAPI(title="Cat vs Dog Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model globally
MODEL_PATH = "best_model_xception.keras"
model = None

try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'best_model_xception.keras' is in the current directory.")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}
    
    try:
        # Read the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Xception model typically expects 299x299 images
        image = image.resize((299, 299))
        
        # Convert to array and preprocess
        img_array = keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.xception.preprocess_input(img_array)
        
        # Predict
        predictions = model.predict(img_array)
        
        # Assuming binary classification where 0=Cat, 1=Dog (or vice versa depending on training)
        # Using a standard 0.5 threshold if it's a single sigmoid output
        if predictions.shape[-1] == 1:
            score = predictions[0][0]
            # Adjust mapping based on how the model was trained; standard is 0=Cat, 1=Dog for alphabetical order
            class_name = "Dog" if score > 0.5 else "Cat"
            confidence = float(score if score > 0.5 else 1.0 - score)
        else:
            # If softmax with 2 outputs
            class_idx = np.argmax(predictions[0])
            score = predictions[0][class_idx]
            classes = ["Cat", "Dog"] # Replace with actual class names from training
            class_name = classes[class_idx]
            confidence = float(score)
            
        return {
            "prediction": class_name,
            "confidence": round(confidence * 100, 2),
            "raw_score": float(predictions[0][0]) if predictions.shape[-1] == 1 else float(score)
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
