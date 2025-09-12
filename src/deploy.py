from PIL import Image

import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify

from model import AbstractDetector


#Déploiement du serveur avec Flask et chargement du modèle
app = Flask(__name__)

model = AbstractDetector()
model.load_state_dict(torch.load("models/abstract_cnn.pth"))
model.eval()

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

@app.route("/predict", methods=["POST"])
def predict():
    """Route de prédiction d'une image reçue par requête POST
    Exemple : curl -X POST -F "file=@$FILE" http://127.0.0.1:5000/predict
    """

    if "file" not in request.files:
        return jsonify({"error": "Pas d'image envoyée"}), 400

    file = request.files["file"]
    try:
        img = Image.open(file.stream).convert("RGB")
    except:
        return jsonify({"error": "Format d'image invalide"}), 400

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        prob = output.item()
        label = 1 if prob >= 0.5 else 0

    return jsonify({
        "probability_abstract": prob,
        "predicted_label": "abstract" if label == 1 else "photo"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
