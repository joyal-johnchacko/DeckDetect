from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import json
import io
from model import CardClassifier

# ============================================
# 1. LOAD CLASS NAMES & MODEL
# ============================================
app = Flask(__name__)

with open("class_names.json", "r") as f:
    class_names = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CardClassifier(num_classes=len(class_names))
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print(f"Model loaded! {len(class_names)} classes ready.")


# ============================================
# 2. IMAGE TRANSFORM
# ============================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ============================================
# 3. PREDICT FUNCTION
# ============================================
def predict_card(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top5_probs, top5_indices = torch.topk(probabilities, 5)

    results = []
    for prob, idx in zip(top5_probs, top5_indices):
        results.append({
            "card": class_names[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return results


# ============================================
# 4. ROUTES
# ============================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        results = predict_card(image_bytes)
        return jsonify({"predictions": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# 5. RUN SERVER
# ============================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)