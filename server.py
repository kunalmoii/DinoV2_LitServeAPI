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
import numpy as np
import cv2
import json


class DinoClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, x):
        features = self.backbone(x).last_hidden_state[:, 0]
        return self.classifier(features)


class DinoLitAPI(ls.LitAPI):
    def setup(self, device):
        # Load processor & backbone
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        backbone = AutoModel.from_pretrained("facebook/dinov2-base")

        # Load class info & classifier
        self.class_names = sorted(["Empty", "Object", "Occlusion"])
        num_classes = len(self.class_names)
        self.model = DinoClassifier(backbone, num_classes)
        self.model.load_state_dict(
            torch.load("checkpoints/best_model.pth", map_location=device)
        )
        self.model.to(device)
        self.model.eval()

        # Load bay coordinates once
        with open("utils/bay_coordinates.json", "r") as f:
            self.bay_coordinates = json.load(f)

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
        self.original_image = np.array(image)
        return self.original_image

    def predict(self, image_np: np.ndarray):
        return self._predict_per_bay(image_np, self.bay_coordinates)

    def _predict_per_bay(self, image: np.ndarray, bay_coordinates: dict):
        h, w = image.shape[:2]
        print(h, w)
        results = []
        original_size = (1920, 1080)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # # Scaling factors
        x_scale = w / original_size[0]
        y_scale = h / original_size[1]

        for bay_name, coords in bay_coordinates.items():
            if not coords:
                continue

            # Ensure closed polygon
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            scaled_coords = [[int(x * x_scale), int(y * y_scale)] for x, y in coords]
            polygon = np.array(scaled_coords, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)

            masked = cv2.bitwise_and(image, image, mask=mask)
            x, y, box_w, box_h = cv2.boundingRect(polygon)
            x_end, y_end = min(x + box_w, w), min(y + box_h, h)

            if x >= w or y >= h or x_end <= x or y_end <= y:
                print(f"⚠️ Skipping invalid crop for {bay_name}")
                continue

            crop = masked[y:y_end, x:x_end]

            if crop.size == 0:
                continue

            # Inference
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            # Save the image to disk
            os.makedirs("crop", exist_ok=True)
            pil_crop.save(f"crop/{bay_name}.jpg")
            image_tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)

            results.append(
                {
                    "bay": bay_name,
                    "prediction": self.class_names[pred_idx.item()],
                    "confidence": round(confidence.item(), 4),
                }
            )

        return results

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    server = ls.LitServer(DinoLitAPI(), accelerator="auto", max_batch_size=1)
    server.run(port=8000)
