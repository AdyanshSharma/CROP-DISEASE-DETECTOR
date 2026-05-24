import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import json
from PIL import Image
from torchvision import transforms
from src.model import get_model

# Config
MODEL_PATH = "models/best_model.pth"
CLASSES_PATH = "models/classes.json"
IMG_SIZE = 224

@st.cache_resource
def load_model_and_classes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
        
    model = get_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    return model, classes, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_image(image, model, classes, device):
    img_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_class_idx = torch.max(probabilities, 0)
    predicted_class = classes[top_class_idx.item()]
    confidence = top_prob.item() * 100

    return predicted_class, confidence, probabilities

st.set_page_config(page_title="Crop Disease Detector", page_icon="🍃")

st.title("🍃 Crop Disease Detector")
st.markdown("Upload a picture of a crop leaf to detect possible diseases.")

try:
    model, classes, device = load_model_and_classes()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Detecting...")
    predicted_class, confidence, probabilities = predict_image(image, model, classes, device)
    
    # Clean up the class name for display
    display_name = predicted_class.replace("_", " ").strip()
    
    st.success(f"**Prediction:** {display_name}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    
    st.subheader("Top 3 Predictions")
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    for i in range(3):
        cls_name = classes[top3_indices[i].item()].replace("_", " ").strip()
        st.write(f"{i+1}. **{cls_name}** ({top3_prob[i].item()*100:.2f}%)")
