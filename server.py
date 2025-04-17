# server.py
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import litserve as ls
import io
import base64
import os


# === Define the classifier (same as your model.py) ===
class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, x):
        features = self.backbone(x).last_hidden_state[:, 0]
        return self.classifier(features)


# === Define LitServe API ===
class DinoLitAPI(ls.LitAPI):
    def setup(self, device):
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        backbone = AutoModel.from_pretrained("facebook/dinov2-base")
        self.model = DinoClassifier(backbone, num_classes=3)
        self.model.to(device)

        # Load model weights
        model_path = "checkpoints/best_model.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print("✅ Model weights loaded.")

        # Set class names
        self.class_names = sorted(["EMPTY", "Object", "Occlusion"])
        num_classes = len(self.class_names)

        # Load the trained classifier
        self.model = DinoClassifier(backbone, num_classes)
        self.model.load_state_dict(
            torch.load("checkpoints/best_model.pth", map_location=device)
        )
        self.model.to(device)
        self.model.eval()

        # Define transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.processor.image_mean, std=self.processor.image_std
                ),
            ]
        )

    def decode_request(self, request):
        image_bytes = base64.b64decode(request["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.processor(image, return_tensors="pt")["pixel_values"]
        return image_tensor


    def predict(self, x):
        device = next(self.model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        class_name = self.class_names[pred_idx.item()]
        confidence_score = round(confidence.item(), 4)  # rounded for readability

        return {"prediction": class_name, "confidence": confidence_score}

    def encode_response(self, output):
        return output


# === Run the LitServe Server ===
if __name__ == "__main__":
    server = ls.LitServer(DinoLitAPI(), accelerator="auto", max_batch_size=1)
    server.run(port=8000)
