from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import base64
import io
import requests
import os
import tempfile

# URL de téléchargement direct OneDrive (modifiée)
model_url = "https://onedrive.live.com/download?cid=0D4B5A54C6A73CFF&resid=0D4B5A54C6A73CFF%211261&authkey=ANuKYnFiJe_wUbA"

# Définir les noms de classes
class_names = {0: 'bottomwear', 1: 'shoes', 2: 'topwear'}

# Créer une instance FastAPI
app = FastAPI()

# Message d'accueil pour l'API pour montrer que tout fonctionne correctement
@app.get("/")
async def read_root():
    return {"message": "Bienvenue sur l'API de classification d'images de vêtements."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Vérifier le format du fichier
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format de fichier non pris en charge. Utilisez JPEG ou PNG.")
    
    # Téléchargement temporaire du modèle depuis OneDrive
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_model_file:
        response = requests.get(model_url)
        if response.status_code == 200:
            temp_model_file.write(response.content)
        else:
            raise HTTPException(status_code=500, detail="Erreur lors du téléchargement du modèle.")

    # Charger le modèle téléchargé temporairement
    model = load_model(temp_model_file.name)

    # Charger et prétraiter l'image
    image = Image.open(file.file).convert("RGB")
    image = image.resize((64, 64))  # Adapter à la taille d'entrée du modèle
    image_array = img_to_array(image) / 255.0  # Normaliser l'image
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension pour le batch
    
    # Prédire la catégorie
    pred_probs = model.predict(image_array)
    pred_label_idx = np.argmax(pred_probs, axis=1)[0]
    pred_class = class_names[pred_label_idx]
    
    # Encoder l'image en base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Supprimer le fichier modèle temporaire
    os.remove(temp_model_file.name)
    
    return {
        "predicted_class": pred_class,
        "predicted_probability": float(np.max(pred_probs)),
        "image_base64": img_str
    }