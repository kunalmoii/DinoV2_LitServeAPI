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
from mqtt_client import BayMQTTPublisher
from dotenv import load_dotenv

load_dotenv()


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

        # Load processor with error handling
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                "models/dinov2-base", local_files_only=True
            )
        except Exception as e:
            print(f"Failed to load processor from local files: {e}")
            print("Attempting to download processor...")
            self.processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base", force_download=True
            )

        # Load backbone with error handling and validation
        try:
            # Check if model files exist and are valid
            model_path = "models/dinov2-base"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model directory {model_path} not found")

            # Check for safetensors file and validate size
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                file_size = os.path.getsize(safetensors_path)
                print(f"Safetensors file size: {file_size / (1024**2):.2f} MB")
                # If file is suspiciously small (< 10MB), it's likely corrupted
                if file_size < 10 * 1024 * 1024:
                    raise ValueError(
                        f"Safetensors file appears corrupted (size: {file_size} bytes)"
                    )

            backbone = AutoModel.from_pretrained(
                "models/dinov2-base", local_files_only=True
            )
            print("Successfully loaded backbone from local files")

        except Exception as e:
            print(f"Failed to load backbone from local files: {e}")
            print("Attempting to download backbone model...")
            try:
                backbone = AutoModel.from_pretrained(
                    "facebook/dinov2-base", force_download=True
                )
                print("Successfully downloaded and loaded backbone")
                # Save the model locally for future use
                os.makedirs("models/dinov2-base", exist_ok=True)
                backbone.save_pretrained("models/dinov2-base")
                print("Saved backbone model locally")
            except Exception as download_error:
                print(f"Failed to download backbone: {download_error}")
                raise RuntimeError(
                    "Unable to load backbone model from local files or download"
                )

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

        # Initialize MQTT publisher only if enabled
        self.mqtt_enabled = os.getenv("ENABLE_MQTT", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        if self.mqtt_enabled:
            self.mqtt_publisher = BayMQTTPublisher()
        else:
            self.mqtt_publisher = None

    def decode_request(self, request):
        image_bytes = base64.b64decode(request["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        self.original_image = np.array(image)
        return self.original_image

    def predict(self, image_np: np.ndarray):
        results = self._predict_per_bay(image_np, self.bay_coordinates)
        # Publish results to MQTT only if enabled
        if self.mqtt_enabled and self.mqtt_publisher:
            self.mqtt_publisher.publish_bay_results(results)
        return results

    def _predict_per_bay(self, image: np.ndarray, bay_coordinates: dict):
        h, w = image.shape[:2]
        print(h, w)
        results = []
        original_size = (1920, 1080)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        x_scale = w / original_size[0]
        y_scale = h / original_size[1]

        image_tensors = []
        bay_names = []

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

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            os.makedirs("crop", exist_ok=True)
            pil_crop.save(f"crop/{bay_name}.jpg")

            image_tensor = self.transform(pil_crop).unsqueeze(0)
            image_tensors.append(image_tensor)
            bay_names.append(bay_name)

        if not image_tensors:
            return results

        # Run batched inference
        batch = torch.cat(image_tensors).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            confidences, pred_indices = torch.max(probs, dim=1)

        for i, bay_name in enumerate(bay_names):
            results.append(
                {
                    "bay": bay_name,
                    "prediction": self.class_names[pred_indices[i].item()],
                    "confidence": round(confidences[i].item(), 4),
                }
            )

        return results

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    server = ls.LitServer(DinoLitAPI(), accelerator="auto", max_batch_size=1)
    server.run(port=8000)
