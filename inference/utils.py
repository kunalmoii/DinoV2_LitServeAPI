# inference/utils.py

import os
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# === Load Processor & Define Transforms ===
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

infer_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ]
)


# === Classifier Definition ===
class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, x):
        features = self.backbone(x).last_hidden_state[:, 0]
        return self.classifier(features)


# === Utility Functions ===
def load_model(model_path, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = AutoModel.from_pretrained("facebook/dinov2-base")
    model = DinoClassifier(backbone, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_image(model, device, image_path, class_names):
    image = Image.open(image_path).convert("RGB")
    image_tensor = infer_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    return class_names[predicted_class]
